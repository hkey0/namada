//! Primitives that facilitate keeping track of the number
//! of bytes utilized by some Tendermint consensus round's proposal.
//!
//! This is important, because Tendermint places an upper bound
//! on the size of a block, rejecting blocks whose size exceeds
//! the limit stated in [`RequestPrepareProposal`].
//!
//! The code in this module doesn't perform any deserializing to
//! verify if we are, in fact, allocating space for the correct
//! kind of tx for the current [`BlockAllocator`] state. It
//! is up to `PrepareProposal` to dispatch the correct kind of tx
//! into the current state of the allocator.
//!
//! # How space is allocated
//!
//! In the current implementation, we allocate space for transactions
//! in the following order of preference:
//!
//! - First, we allot space for protocol txs. We allow them to take up at most
//!   1/2 of the total block space unless there is extra room due to a lack of
//!   user txs.
//! - Next, we allot space for user submitted txs until the block is filled.
//! - If we cannot fill the block with normal txs, we try to fill it with
//!   protocol txs that were not allocated in the initial phase.
//!
//!
//! # How gas is allocated
//!
//! Gas is only relevant to non-protocol txs. Every such tx defines its
//! gas limit. We take this entire gas limit as the amount of gas requested by
//! the tx.

pub mod states;

// TODO: what if a tx has a size greater than the threshold for
// its bin? how do we handle this? if we keep it in the mempool
// forever, it'll be a DoS vec, as we can make nodes run out of
// memory! maybe we should allow block decisions for txs that are
// too big to fit in their respective bin? in these special block
// decisions, we would only decide proposals with "large" txs??
//
// MAYBE: in the state machine impl, reset to beginning state, and
// and alloc space for large tx right at the start. the problem with
// this is that then we may not have enough space for decrypted txs

use std::marker::PhantomData;

use namada::proof_of_stake::pos_queries::PosQueries;
use namada::state::{self, WlStorage};

#[allow(unused_imports)]
use crate::facade::tendermint_proto::abci::RequestPrepareProposal;
use crate::node::ledger::shell::block_alloc::states::WithNormalTxs;

/// Block allocation failure status responses.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum AllocFailure {
    /// The transaction can only be included in an upcoming block.
    ///
    /// We return the resource left in the tx bin for logging purposes.
    Rejected { bin_resource_left: u64 },
    /// The transaction would overflow the allotted bin resource,
    /// therefore it needs to be handled separately.
    ///
    /// We return the resource allotted to the tx bin for logging purposes.
    OverflowsBin { bin_resource: u64 },
}

/// The block resources that need to be allocated
pub struct BlockResources<'tx> {
    tx: &'tx [u8],
    gas: u64,
}

impl<'tx> BlockResources<'tx> {
    /// Generates a new block resource instance
    pub fn new(tx: &'tx [u8], gas: u64) -> Self {
        Self { tx, gas }
    }
}

/// Marker type for the block space
#[derive(Debug, Default, Clone, Copy)]
pub struct BlockSpace;
/// Marker type for the block gas
#[derive(Debug, Default, Clone, Copy)]
pub struct BlockGas;

pub trait Resource {
    type Input<'r>;

    fn usage_of(input: Self::Input<'_>) -> u64;
}

impl Resource for BlockSpace {
    type Input<'r> = &'r [u8];

    fn usage_of(input: Self::Input<'_>) -> u64 {
        input.len() as u64
    }
}

impl Resource for BlockGas {
    type Input<'r> = u64;

    fn usage_of(input: Self::Input<'_>) -> u64 {
        input
    }
}

/// Allotted resources for a batch of transactions in some proposed block.
///
/// We keep track of the current space utilized by:
///
///   - normal transactions.
///   - Protocol transactions.
///
/// Gas usage of normal txs is also tracked.
#[derive(Debug, Default)]
pub struct BlockAllocator<State> {
    /// The current state of the [`BlockAllocator`] state machine.
    _state: PhantomData<*const State>,
    /// The total space Tendermint has allotted to the
    /// application for the current block height.
    block: TxBin<BlockSpace>,
    /// The current space utilized by protocol transactions.
    protocol_txs: TxBin<BlockSpace>,
    /// The current space and gas utilized by normal user transactions.
    normal_txs: NormalTxsBins,
}

impl<D, H> From<&WlStorage<D, H>>
    for BlockAllocator<states::BuildingProtocolTxBatch<WithNormalTxs>>
where
    D: 'static + state::DB + for<'iter> state::DBIter<'iter>,
    H: 'static + state::StorageHasher,
{
    #[inline]
    fn from(storage: &WlStorage<D, H>) -> Self {
        Self::init(
            storage.pos_queries().get_max_proposal_bytes().get(),
            namada::parameters::get_max_block_gas(storage).unwrap(),
        )
    }
}

impl BlockAllocator<states::BuildingProtocolTxBatch<WithNormalTxs>> {
    /// Construct a new [`BlockAllocator`], with an upper bound
    /// on the max size of all txs in a block defined by Tendermint and an upper
    /// bound on the max gas in a block.
    #[inline]
    pub fn init(
        tendermint_max_block_space_in_bytes: u64,
        max_block_gas: u64,
    ) -> Self {
        let max = tendermint_max_block_space_in_bytes;
        Self {
            _state: PhantomData,
            block: TxBin::init(max),
            protocol_txs: {
                let allotted_space_in_bytes = threshold::ONE_HALF.over(max);
                TxBin::init(allotted_space_in_bytes)
            },
            normal_txs: NormalTxsBins::new(max_block_gas),
        }
    }
}

impl BlockAllocator<states::BuildingTxBatch> {
    /// Construct a new [`BlockAllocator`], with an upper bound
    /// on the max size of all txs in a block defined by Tendermint and an upper
    /// bound on the max gas in a block.
    #[inline]
    pub fn init(
        tendermint_max_block_space_in_bytes: u64,
        max_block_gas: u64,
    ) -> Self {
        let max = tendermint_max_block_space_in_bytes;
        Self {
            _state: PhantomData,
            block: TxBin::init(max),
            protocol_txs: TxBin::default(),
            normal_txs: NormalTxsBins::new(max_block_gas),
        }
    }
}

impl<State> BlockAllocator<State> {
    /// Return the amount of space left to initialize in all
    /// [`TxBin`] instances.
    ///
    /// This is calculated based on the difference between the Tendermint
    /// block space for a given round and the sum of the allotted space
    /// to each [`TxBin`] instance in a [`BlockAllocator`].
    #[inline]
    fn uninitialized_space_in_bytes(&self) -> u64 {
        let total_bin_space =
            self.protocol_txs.allotted + self.normal_txs.space.allotted;
        self.block.allotted - total_bin_space
    }
}

/// Allotted resource for a batch of transactions of the same kind in some
/// proposed block. At the moment this is used to track two resources of the
/// block: space and gas. Space is measured in bytes while gas in gas units.
#[derive(Debug, Copy, Clone, Default)]
pub struct TxBin<R: Resource> {
    /// The current resource utilization of the batch of transactions.
    occupied: u64,
    /// The maximum resource amount the batch of transactions may occupy.
    allotted: u64,
    /// The resource that this bin is tracking
    _resource: PhantomData<R>,
}

impl<R: Resource> TxBin<R> {
    /// Return the amount of resource left in this [`TxBin`].
    #[inline]
    pub fn resource_left(&self) -> u64 {
        self.allotted - self.occupied
    }

    /// Construct a new [`TxBin`], with a capacity of `max_capacity`.
    #[inline]
    pub fn init(max_capacity: u64) -> Self {
        Self {
            allotted: max_capacity,
            occupied: 0,
            _resource: PhantomData,
        }
    }

    /// Shrink the allotted resource of this [`TxBin`] to whatever
    /// amount is currently being utilized.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.allotted = self.occupied;
    }

    /// Try to dump a new transaction into this [`TxBin`].
    ///
    /// Signal the caller if the tx requires more resource than its max
    /// allotted.
    pub fn try_dump(
        &mut self,
        resource: R::Input<'_>,
    ) -> Result<(), AllocFailure> {
        let resource = R::usage_of(resource);
        if resource > self.allotted {
            let bin_size = self.allotted;
            return Err(AllocFailure::OverflowsBin {
                bin_resource: bin_size,
            });
        }
        let occupied = self.occupied + resource;
        if occupied <= self.allotted {
            self.occupied = occupied;
            Ok(())
        } else {
            let bin_resource_left = self.resource_left();
            Err(AllocFailure::Rejected { bin_resource_left })
        }
    }
}

#[derive(Debug, Default)]
pub struct NormalTxsBins {
    space: TxBin<BlockSpace>,
    gas: TxBin<BlockGas>,
}

impl NormalTxsBins {
    pub fn new(max_gas: u64) -> Self {
        Self {
            space: TxBin::default(),
            gas: TxBin::init(max_gas),
        }
    }

    pub fn try_dump(&mut self, tx: &[u8], gas: u64) -> Result<(), String> {
        self.space.try_dump(tx).map_err(|e| match e {
            AllocFailure::Rejected { .. } => {
                "No more space left in the block for normal txs".to_string()
            }
            AllocFailure::OverflowsBin { .. } => "The given wrapper tx is \
                                                  larger than the remaining \
                                                  available block space"
                .to_string(),
        })?;
        self.gas.try_dump(gas).map_err(|e| match e {
            AllocFailure::Rejected { .. } => {
                "No more gas left in the block for wrapper txs".to_string()
            }
            AllocFailure::OverflowsBin { .. } => {
                "The given wrapper tx requires more gas than available to the \
                 entire block"
                    .to_string()
            }
        })
    }
}

pub mod threshold {
    //! Transaction allotment thresholds.

    use num_rational::Ratio;

    /// Threshold over a portion of block space.
    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct Threshold(Ratio<u64>);

    impl Threshold {
        /// Return a new [`Threshold`].
        const fn new(numer: u64, denom: u64) -> Self {
            // constrain ratio to a max of 1
            let numer = if numer > denom { denom } else { numer };
            Self(Ratio::new_raw(numer, denom))
        }

        /// Return a [`Threshold`] over some free space.
        pub fn over(self, free_space_in_bytes: u64) -> u64 {
            (self.0 * free_space_in_bytes).to_integer()
        }
    }

    /// Divide free space in three.
    pub const ONE_HALF: Threshold = Threshold::new(1, 2);
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use assert_matches::assert_matches;
    use proptest::prelude::*;

    use super::states::{
        BuildingProtocolTxBatch, BuildingTxBatch, NextState, TryAlloc,
    };
    use super::*;
    use crate::node::ledger::shims::abcipp_shim_types::shim::TxBytes;

    /// Convenience alias for a block space allocator at a state with protocol
    /// txs.
    type BsaInitialProtocolTxs =
        BlockAllocator<BuildingProtocolTxBatch<WithNormalTxs>>;

    /// Convenience alias for a block space allocator at a state with protocol
    /// txs.
    type BsaNormalTxs = BlockAllocator<BuildingTxBatch>;

    /// Proptest generated txs.
    #[derive(Debug)]
    struct PropTx {
        tendermint_max_block_space_in_bytes: u64,
        max_block_gas: u64,
        protocol_txs: Vec<TxBytes>,
        normal_txs: Vec<TxBytes>,
    }

    /// Check that at most 1/2 of the block space is
    /// reserved for each kind of tx type, in the
    /// allocator's common path. Further check that
    /// if not enough normal txs are present, the rest
    /// if filled with protocol txs
    #[test]
    fn test_filling_up_with_protocol() {
        const BLOCK_SIZE: u64 = 60;
        const BLOCK_GAS: u64 = 1_000;

        // reserve block space for protocol txs
        let mut alloc = BsaInitialProtocolTxs::init(BLOCK_SIZE, BLOCK_GAS);

        // allocate ~1/2 of the block space to encrypted txs
        assert!(alloc.try_alloc(&[0; 29]).is_ok());

        // reserve block space for normal txs
        let mut alloc = alloc.next_state();

        // the space we allotted to protocol txs was shrunk to
        // the total space we actually used up
        assert_eq!(alloc.protocol_txs.allotted, 29);

        // check that the allotted space for normal txs is correct
        assert_eq!(alloc.normal_txs.space.allotted, BLOCK_SIZE - 29);

        // add about ~1/3 worth of normal txs
        assert!(alloc.try_alloc(BlockResources::new(&[0; 17], 0)).is_ok());

        // fill the rest of the block with protocol txs
        let mut alloc = alloc.next_state();

        // check that space was shrunk
        assert_eq!(alloc.protocol_txs.allotted, BLOCK_SIZE - (29 + 17));

        // add protocol txs to the block space allocator
        assert!(alloc.try_alloc(&[0; 14]).is_ok());

        // the block should be full at this point
        assert_matches!(
            alloc.try_alloc(&[0; 1]),
            Err(AllocFailure::Rejected { .. })
        );
    }

    /// Test that if less than half of the block can be initially filled
    /// with protocol txs, the rest if filled with normal txs.
    #[test]
    fn test_less_than_half_protocol() {
        const BLOCK_SIZE: u64 = 60;
        const BLOCK_GAS: u64 = 1_000;

        // reserve block space for protocol txs
        let mut alloc = BsaInitialProtocolTxs::init(BLOCK_SIZE, BLOCK_GAS);

        // allocate ~1/3 of the block space to encrypted txs
        assert!(alloc.try_alloc(&[0; 18]).is_ok());

        // reserve block space for normal txs
        let mut alloc = alloc.next_state();

        // the space we allotted to protocol txs was shrunk to
        // the total space we actually used up
        assert_eq!(alloc.protocol_txs.allotted, 18);

        // check that the allotted space for normal txs is correct
        assert_eq!(alloc.normal_txs.space.allotted, BLOCK_SIZE - 18);

        // add about ~2/3 worth of normal txs
        assert!(alloc.try_alloc(BlockResources::new(&[0; 42], 0)).is_ok());
        // the block should be full at this point
        assert_matches!(
            alloc.try_alloc(BlockResources::new(&[0; 1], 0)),
            Err(AllocFailure::Rejected { .. })
        );
    }

    proptest! {
        /// Check if we reject a tx when its respective bin
        /// capacity has been reached on a [`BlockAllocator`].
        #[test]
        fn test_reject_tx_on_bin_cap_reached(max in prop::num::u64::ANY) {
            proptest_reject_tx_on_bin_cap_reached(max)
        }

        /// Check if the initial bin capacity of the [`BlockAllocator`]
        /// is correct.
        #[test]
        fn test_initial_bin_capacity(max in prop::num::u64::ANY) {
            proptest_initial_bin_capacity(max)
        }

        /// Test that dumping txs whose total combined size
        /// is less than the bin cap does not fill up the bin.
        #[test]
        fn test_tx_dump_doesnt_fill_up_bin(args in arb_transactions()) {
            proptest_tx_dump_doesnt_fill_up_bin(args)
        }
    }

    /// Implementation of [`test_reject_tx_on_bin_cap_reached`].
    fn proptest_reject_tx_on_bin_cap_reached(
        tendermint_max_block_space_in_bytes: u64,
    ) {
        let mut bins =
            BsaNormalTxs::init(tendermint_max_block_space_in_bytes, 1_000);

        // fill the entire bin of protocol txs
        bins.normal_txs.space.occupied = bins.normal_txs.space.allotted;

        // make sure we can't dump any new protocol txs in the bin
        assert_matches!(
            bins.try_alloc(BlockResources::new(b"arbitrary tx bytes", 0)),
            Err(AllocFailure::Rejected { .. })
        );

        // Reset space bin
        bins.normal_txs.space.occupied = 0;
        // Fill the entire gas bin
        bins.normal_txs.gas.occupied = bins.normal_txs.gas.allotted;

        // Make sure we can't dump any new wncrypted txs in the bin
        assert_matches!(
            bins.try_alloc(BlockResources::new(b"arbitrary tx bytes", 1)),
            Err(AllocFailure::Rejected { .. })
        )
    }

    /// Implementation of [`test_initial_bin_capacity`].
    fn proptest_initial_bin_capacity(tendermint_max_block_space_in_bytes: u64) {
        let bins = BsaInitialProtocolTxs::init(
            tendermint_max_block_space_in_bytes,
            1_000,
        );
        let expected = tendermint_max_block_space_in_bytes
            - threshold::ONE_HALF.over(tendermint_max_block_space_in_bytes);
        assert_eq!(expected, bins.uninitialized_space_in_bytes());
    }

    /// Implementation of [`test_tx_dump_doesnt_fill_up_bin`].
    fn proptest_tx_dump_doesnt_fill_up_bin(args: PropTx) {
        let PropTx {
            tendermint_max_block_space_in_bytes,
            max_block_gas,
            protocol_txs,
            normal_txs,
        } = args;

        // produce new txs until the moment we would have
        // filled up the bins.
        //
        // iterate over the produced txs to make sure we can keep
        // dumping new txs without filling up the bins

        let bins = RefCell::new(BsaInitialProtocolTxs::init(
            tendermint_max_block_space_in_bytes,
            max_block_gas,
        ));
        let mut protocol_tx_iter = protocol_txs.iter();
        let mut allocated_txs = vec![];
        for tx in protocol_tx_iter.by_ref() {
            let bin = bins.borrow().protocol_txs;
            let new_size = bin.occupied + tx.len() as u64;
            if new_size >= bin.allotted {
                break;
            } else {
                allocated_txs.push(tx);
            }
        }
        for tx in allocated_txs {
            assert!(bins.borrow_mut().try_alloc(tx).is_ok());
        }

        let bins = RefCell::new(bins.into_inner().next_state());
        let decrypted_txs = normal_txs.into_iter().take_while(|tx| {
            let bin = bins.borrow().normal_txs.space;
            let new_size = bin.occupied + tx.len() as u64;
            new_size < bin.allotted
        });
        for tx in decrypted_txs {
            assert!(
                bins.borrow_mut()
                    .try_alloc(BlockResources::new(&tx, 0))
                    .is_ok()
            );
        }

        let bins = RefCell::new(bins.into_inner().next_state());
        let mut allocated_txs = vec![];
        for tx in protocol_tx_iter.by_ref() {
            let bin = bins.borrow().protocol_txs;
            let new_size = bin.occupied + tx.len() as u64;
            if new_size >= bin.allotted {
                break;
            } else {
                allocated_txs.push(tx);
            }
        }

        for tx in allocated_txs {
            assert!(bins.borrow_mut().try_alloc(tx).is_ok());
        }
    }

    prop_compose! {
        /// Generate arbitrarily sized txs of different kinds.
        fn arb_transactions()
            // create base strategies
            (
                (tendermint_max_block_space_in_bytes, max_block_gas, protocol_tx_max_bin_size,
                 decrypted_tx_max_bin_size) in arb_max_bin_sizes(),
            )
            // compose strategies
            (
                tendermint_max_block_space_in_bytes in Just(tendermint_max_block_space_in_bytes),
            max_block_gas in Just(max_block_gas),
                protocol_txs in arb_tx_list(protocol_tx_max_bin_size),
                normal_txs in arb_tx_list(decrypted_tx_max_bin_size),
            )
            -> PropTx {
                PropTx {
                    tendermint_max_block_space_in_bytes,
                max_block_gas,
                    protocol_txs: protocol_txs.into_iter().map(prost::bytes::Bytes::from).collect(),
                    normal_txs: normal_txs.into_iter().map(prost::bytes::Bytes::from).collect(),
                }
            }
    }

    /// Return random bin sizes for a [`BlockAllocator`].
    fn arb_max_bin_sizes() -> impl Strategy<Value = (u64, u64, usize, usize)> {
        const MAX_BLOCK_SIZE_BYTES: u64 = 1000;
        (1..=MAX_BLOCK_SIZE_BYTES).prop_map(
            |tendermint_max_block_space_in_bytes| {
                (
                    tendermint_max_block_space_in_bytes,
                    tendermint_max_block_space_in_bytes,
                    threshold::ONE_HALF
                        .over(tendermint_max_block_space_in_bytes)
                        as usize,
                    threshold::ONE_HALF
                        .over(tendermint_max_block_space_in_bytes)
                        as usize,
                )
            },
        )
    }

    /// Return a list of txs.
    fn arb_tx_list(max_bin_size: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
        const MAX_TX_NUM: usize = 64;
        let tx = prop::collection::vec(prop::num::u8::ANY, 0..=max_bin_size);
        prop::collection::vec(tx, 0..=MAX_TX_NUM)
    }
}
