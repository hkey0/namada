//! Logic to do with events emitted by the ledger.
pub mod log;

use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{self, Display};
use std::ops::{Index, IndexMut};
use std::str::FromStr;

use borsh::{BorshDeserialize, BorshSerialize};
use namada_core::types::ethereum_structs::{BpTransferStatus, EthBridgeEvent};
use namada_core::types::ibc::IbcEvent;
use namada_tx::data::TxType;
use serde_json::Value;

// use crate::ledger::governance::utils::ProposalEvent;
use crate::error::{EncodingError, Error, EventError};
use crate::tendermint_proto::v0_37::abci::EventAttribute;

impl From<EthBridgeEvent> for Event {
    #[inline]
    fn from(event: EthBridgeEvent) -> Event {
        Self::from(&event)
    }
}

impl From<&EthBridgeEvent> for Event {
    fn from(event: &EthBridgeEvent) -> Event {
        match event {
            EthBridgeEvent::BridgePool { tx_hash, status } => Event {
                event_type: EventType::EthereumBridge,
                level: EventLevel::Tx,
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert(
                        "kind".into(),
                        match status {
                            BpTransferStatus::Relayed => "bridge_pool_relayed",
                            BpTransferStatus::Expired => "bridge_pool_expired",
                        }
                        .into(),
                    );
                    attrs.insert("tx_hash".into(), tx_hash.to_string());
                    attrs
                },
            },
        }
    }
}

/// Indicates if an event is emitted do to
/// an individual Tx or the nature of a finalized block
#[derive(Clone, Debug, Eq, PartialEq, BorshSerialize, BorshDeserialize)]
pub enum EventLevel {
    /// Indicates an event is to do with a finalized block.
    Block,
    /// Indicates an event is to do with an individual transaction.
    Tx,
}

/// Custom events that can be queried from Tendermint
/// using a websocket client
#[derive(Clone, Debug, Eq, PartialEq, BorshSerialize, BorshDeserialize)]
pub struct Event {
    /// The type of event.
    pub event_type: EventType,
    /// The level of the event - whether it relates to a block or an individual
    /// transaction.
    pub level: EventLevel,
    /// Key-value attributes of the event.
    pub attributes: HashMap<String, String>,
}

/// The two types of custom events we currently use
#[derive(Clone, Debug, Eq, PartialEq, BorshSerialize, BorshDeserialize)]
pub enum EventType {
    /// The transaction was applied during block finalization
    Applied,
    /// The IBC transaction was applied during block finalization
    Ibc(String),
    /// The proposal that has been executed
    Proposal,
    /// The pgf payment
    PgfPayment,
    /// Ethereum Bridge event
    EthereumBridge,
}

impl Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventType::Applied => write!(f, "applied"),
            EventType::Ibc(t) => write!(f, "{}", t),
            EventType::Proposal => write!(f, "proposal"),
            EventType::PgfPayment => write!(f, "pgf_payment"),
            EventType::EthereumBridge => write!(f, "ethereum_bridge"),
        }?;
        Ok(())
    }
}

impl FromStr for EventType {
    type Err = EventError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "applied" => Ok(EventType::Applied),
            "proposal" => Ok(EventType::Proposal),
            "pgf_payments" => Ok(EventType::PgfPayment),
            // IBC
            "update_client" => Ok(EventType::Ibc("update_client".to_string())),
            "send_packet" => Ok(EventType::Ibc("send_packet".to_string())),
            "write_acknowledgement" => {
                Ok(EventType::Ibc("write_acknowledgement".to_string()))
            }
            "ethereum_bridge" => Ok(EventType::EthereumBridge),
            _ => Err(EventError::InvalidEventType),
        }
    }
}

impl Event {
    /// Creates a new event with the hash and height of the transaction
    /// already filled in
    pub fn new_tx_event(tx: &namada_tx::Tx, height: u64) -> Self {
        let mut event = match tx.header().tx_type {
            TxType::Wrapper(_) => {
                let mut event = Event {
                    event_type: EventType::Applied,
                    level: EventLevel::Tx,
                    attributes: HashMap::new(),
                };
                event["hash"] = tx.header_hash().to_string();
                event
            }
            TxType::Protocol(_) => {
                let mut event = Event {
                    event_type: EventType::Applied,
                    level: EventLevel::Tx,
                    attributes: HashMap::new(),
                };
                event["hash"] = tx.header_hash().to_string();
                event
            }
            _ => unreachable!(),
        };
        event["height"] = height.to_string();
        event["log"] = "".to_string();
        event
    }

    /// Check if the events keys contains a given string
    pub fn contains_key(&self, key: &str) -> bool {
        self.attributes.contains_key(key)
    }

    /// Get the value corresponding to a given key, if it exists.
    /// Else return None.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.attributes.get(key)
    }
}

impl Index<&str> for Event {
    type Output = String;

    fn index(&self, index: &str) -> &Self::Output {
        &self.attributes[index]
    }
}

impl IndexMut<&str> for Event {
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        if !self.attributes.contains_key(index) {
            self.attributes.insert(String::from(index), String::new());
        }
        self.attributes.get_mut(index).unwrap()
    }
}

impl From<IbcEvent> for Event {
    fn from(ibc_event: IbcEvent) -> Self {
        Self {
            event_type: EventType::Ibc(ibc_event.event_type),
            level: EventLevel::Tx,
            attributes: ibc_event.attributes,
        }
    }
}

/// Convert our custom event into the necessary tendermint proto type
impl From<Event> for crate::tendermint_proto::v0_37::abci::Event {
    fn from(event: Event) -> Self {
        Self {
            r#type: event.event_type.to_string(),
            attributes: event
                .attributes
                .into_iter()
                .map(|(key, value)| EventAttribute {
                    key,
                    value,
                    index: true,
                })
                .collect(),
        }
    }
}

/// A thin wrapper around a HashMap for parsing event JSONs
/// returned in tendermint subscription responses.
#[derive(Debug)]
pub struct Attributes(HashMap<String, String>);

impl Attributes {
    /// Get a reference to the value associated with input key
    pub fn get(&self, key: &str) -> Option<&String> {
        self.0.get(key)
    }

    /// Get ownership of the value associated to the input key
    pub fn take(&mut self, key: &str) -> Option<String> {
        self.0.remove(key)
    }
}

impl TryFrom<&serde_json::Value> for Attributes {
    type Error = Error;

    fn try_from(json: &serde_json::Value) -> Result<Self, Self::Error> {
        let mut attributes = HashMap::new();
        let attrs: Vec<serde_json::Value> = serde_json::from_value(
            json.get("attributes")
                .ok_or(EventError::MissingAttributes)?
                .clone(),
        )
        .map_err(|err| EncodingError::Serde(err.to_string()))?;

        for attr in attrs {
            let key = serde_json::from_value(
                attr.get("key")
                    .ok_or_else(|| {
                        try_decoding_str(&attr, EventError::MissingKey)
                    })?
                    .clone(),
            )
            .map_err(|err| EncodingError::Serde(err.to_string()))?;
            let value = serde_json::from_value(
                attr.get("value")
                    .ok_or_else(|| {
                        try_decoding_str(&attr, EventError::MissingValue)
                    })?
                    .clone(),
            )
            .map_err(|err| EncodingError::Serde(err.to_string()))?;
            attributes.insert(key, value);
        }
        Ok(Attributes(attributes))
    }
}

fn try_decoding_str<F>(attr: &Value, err_type: F) -> Error
where
    F: FnOnce(String) -> EventError,
{
    match serde_json::to_string(attr) {
        Ok(e) => Error::from(err_type(e)),
        Err(err) => Error::from(EncodingError::Serde(format!(
            "Failure to decode attribute {}",
            err
        ))),
    }
}
