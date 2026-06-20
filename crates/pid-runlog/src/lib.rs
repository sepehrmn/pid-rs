#![doc = include_str!("../README.md")]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

pub const RUN_LOG_SCHEMA_VERSION: u32 = 1;

pub const RUN_LOG_EVENT_TYPES: &[&str] = &[
    "run_started",
    "run_ended",
    "config_logged",
    "frame_observed",
    "embedding_captured",
    "embedding_contract",
    "sim_snapshot",
    "bridge_request",
    "bridge_response",
    "action_applied",
    "object_pose",
    "flow_gt",
    "flow_pred",
    "pid_metric",
    "geometry_metric",
    "evaluation_metric",
    "label_observed",
    "intervention_applied",
    "artifact_logged",
    "attribution_logged",
    "error_logged",
];

pub const RUN_LOG_SIDECARS: &[&str] = &["validation", "summary", "manifest"];

pub const RUN_LOG_VALIDATION_RULES: &[&str] = &[
    "run log is nonempty",
    "exactly one run_started event",
    "exactly one run_ended event",
    "run_started is first event",
    "run_ended is last event",
    "schema_version matches RUN_LOG_SCHEMA_VERSION",
    "timestamps are nondecreasing",
    "steps are nondecreasing",
    "run_id is nonempty and consistent",
    "config_hash values match canonical config JSON and run_started when config is logged",
    "payload_hash values match canonical payload JSON",
    "bridge request_id values are nonempty and unique",
    "bridge responses refer to existing requests",
    "poses, velocities, flows, and metrics are finite",
    "artifact, embedding, contract, metric, label, and flow source names are nonempty",
    "embedding contract variables have nonempty variable/source names and positive dims",
    "label values are non-null",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActorType {
    HumanGui,
    Script,
    LlmTool,
    System,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Actor {
    pub actor_type: ActorType,
    pub actor_id: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Succeeded,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub position: [f64; 3],
    pub orientation_xyzw: [f64; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimObjectSnapshot {
    pub object_id: String,
    pub pose: Pose,
    pub velocity: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingVariableContract {
    pub variable: String,
    pub source: String,
    pub dims: Vec<usize>,
    pub artifact_uri: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunLogEvent {
    RunStarted {
        schema_version: u32,
        run_id: String,
        timestamp_ns: u64,
        config_hash: String,
        metadata: BTreeMap<String, String>,
    },
    RunEnded {
        run_id: String,
        timestamp_ns: u64,
        status: RunStatus,
        message: Option<String>,
    },
    ConfigLogged {
        timestamp_ns: u64,
        config_hash: String,
        config: serde_json::Value,
    },
    FrameObserved {
        step: u64,
        timestamp_ns: u64,
        observation_hash: Option<String>,
        metadata: BTreeMap<String, String>,
    },
    EmbeddingCaptured {
        step: u64,
        timestamp_ns: u64,
        name: String,
        dims: Vec<usize>,
        artifact_uri: Option<String>,
        sha256: Option<String>,
        metadata: BTreeMap<String, String>,
    },
    EmbeddingContract {
        timestamp_ns: u64,
        name: String,
        variables: Vec<EmbeddingVariableContract>,
        metadata: BTreeMap<String, String>,
    },
    SimSnapshot {
        step: u64,
        timestamp_ns: u64,
        objects: Vec<SimObjectSnapshot>,
        metadata: BTreeMap<String, String>,
    },
    BridgeRequest {
        step: Option<u64>,
        timestamp_ns: u64,
        request_id: String,
        actor: Actor,
        method: String,
        payload_hash: String,
        payload: serde_json::Value,
    },
    BridgeResponse {
        step: Option<u64>,
        timestamp_ns: u64,
        request_id: String,
        ok: bool,
        message: Option<String>,
        result_hash: Option<String>,
    },
    ActionApplied {
        step: u64,
        timestamp_ns: u64,
        actor: Actor,
        action_type: String,
        payload_hash: String,
        payload: serde_json::Value,
    },
    ObjectPose {
        step: u64,
        timestamp_ns: u64,
        object_id: String,
        pose: Pose,
    },
    FlowGt {
        step: u64,
        timestamp_ns: u64,
        object_id: String,
        flow: Vec<[f64; 3]>,
    },
    FlowPred {
        step: u64,
        timestamp_ns: u64,
        source: String,
        object_id: String,
        horizon_steps: u64,
        flow: Vec<[f64; 3]>,
        metadata: BTreeMap<String, String>,
    },
    PidMetric {
        step: u64,
        timestamp_ns: u64,
        name: String,
        value: f64,
        metadata: BTreeMap<String, String>,
    },
    GeometryMetric {
        step: u64,
        timestamp_ns: u64,
        name: String,
        value: f64,
        metadata: BTreeMap<String, String>,
    },
    EvaluationMetric {
        step: u64,
        timestamp_ns: u64,
        name: String,
        value: f64,
        metadata: BTreeMap<String, String>,
    },
    LabelObserved {
        step: u64,
        timestamp_ns: u64,
        name: String,
        value: serde_json::Value,
        metadata: BTreeMap<String, String>,
    },
    InterventionApplied {
        step: u64,
        timestamp_ns: u64,
        actor: Actor,
        intervention_type: String,
        payload_hash: String,
        payload: serde_json::Value,
    },
    ArtifactLogged {
        timestamp_ns: u64,
        name: String,
        kind: String,
        uri: String,
        sha256: Option<String>,
        metadata: BTreeMap<String, String>,
    },
    AttributionLogged {
        timestamp_ns: u64,
        method: String,
        target_output: String,
        layer: Option<String>,
        modality: Option<String>,
        baseline: Option<String>,
        score_hash: Option<String>,
        faithfulness_check: Option<bool>,
        artifact_uri: Option<String>,
        metadata: BTreeMap<String, String>,
    },
    ErrorLogged {
        step: Option<u64>,
        timestamp_ns: u64,
        message: String,
        recoverable: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunLogEventContract {
    pub event_type: String,
    pub has_step: bool,
    pub carries_payload_hash: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunLogContract {
    pub schema_version: u32,
    pub event_types: Vec<RunLogEventContract>,
    pub actor_types: Vec<String>,
    pub run_statuses: Vec<String>,
    pub sidecars: Vec<String>,
    pub validation_rules: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseRecord {
    pub step: u64,
    pub timestamp_ns: u64,
    pub pose: Pose,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricRecord {
    pub step: u64,
    pub timestamp_ns: u64,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionRecord {
    pub step: u64,
    pub timestamp_ns: u64,
    pub actor: Actor,
    pub action_type: String,
    pub payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub step: u64,
    pub timestamp_ns: u64,
    pub name: String,
    pub dims: Vec<usize>,
    pub artifact_uri: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingContractRecord {
    pub timestamp_ns: u64,
    pub name: String,
    pub variables: Vec<EmbeddingVariableContract>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BridgeRecord {
    pub step: Option<u64>,
    pub timestamp_ns: u64,
    pub request_id: String,
    pub method: String,
    pub payload_hash: Option<String>,
    pub ok: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionRecord {
    pub step: u64,
    pub timestamp_ns: u64,
    pub actor: Actor,
    pub intervention_type: String,
    pub payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub timestamp_ns: u64,
    pub name: String,
    pub kind: String,
    pub uri: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionRecord {
    pub timestamp_ns: u64,
    pub method: String,
    pub target_output: String,
    pub layer: Option<String>,
    pub modality: Option<String>,
    pub baseline: Option<String>,
    pub score_hash: Option<String>,
    pub faithfulness_check: Option<bool>,
    pub artifact_uri: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelRecord {
    pub step: u64,
    pub timestamp_ns: u64,
    pub name: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReplayState {
    pub schema_version: Option<u32>,
    pub run_id: Option<String>,
    pub config_hash: Option<String>,
    pub status: Option<RunStatus>,
    pub last_step: Option<u64>,
    pub last_timestamp_ns: Option<u64>,
    pub events_seen: usize,
    pub object_poses: BTreeMap<String, PoseRecord>,
    pub pid_metrics: BTreeMap<String, MetricRecord>,
    pub geometry_metrics: BTreeMap<String, MetricRecord>,
    pub evaluation_metrics: BTreeMap<String, MetricRecord>,
    #[serde(default)]
    pub pid_metric_events: usize,
    #[serde(default)]
    pub geometry_metric_events: usize,
    #[serde(default)]
    pub evaluation_metric_events: usize,
    pub labels: Vec<LabelRecord>,
    pub actions: Vec<ActionRecord>,
    pub interventions: Vec<InterventionRecord>,
    pub artifacts: Vec<ArtifactRecord>,
    pub attributions: Vec<AttributionRecord>,
    pub embeddings: Vec<EmbeddingRecord>,
    pub embedding_contracts: Vec<EmbeddingContractRecord>,
    pub bridge_records: Vec<BridgeRecord>,
    pub sim_snapshots: usize,
    pub errors: Vec<String>,
    pub flow_gt_records: usize,
    pub flow_pred_records: usize,
}

impl ReplayState {
    pub fn apply(&mut self, event: &RunLogEvent) {
        self.events_seen += 1;
        self.last_timestamp_ns = Some(event.timestamp_ns());
        if let Some(step) = event.step() {
            self.last_step = Some(step);
        }

        match event {
            RunLogEvent::RunStarted {
                schema_version,
                run_id,
                config_hash,
                ..
            } => {
                self.schema_version = Some(*schema_version);
                self.run_id = Some(run_id.clone());
                self.config_hash = Some(config_hash.clone());
            }
            RunLogEvent::RunEnded { status, .. } => {
                self.status = Some(status.clone());
            }
            RunLogEvent::ConfigLogged { config_hash, .. } => {
                self.config_hash = Some(config_hash.clone());
            }
            RunLogEvent::FrameObserved { .. } => {}
            RunLogEvent::EmbeddingCaptured {
                step,
                timestamp_ns,
                name,
                dims,
                artifact_uri,
                sha256,
                ..
            } => self.embeddings.push(EmbeddingRecord {
                step: *step,
                timestamp_ns: *timestamp_ns,
                name: name.clone(),
                dims: dims.clone(),
                artifact_uri: artifact_uri.clone(),
                sha256: sha256.clone(),
            }),
            RunLogEvent::EmbeddingContract {
                timestamp_ns,
                name,
                variables,
                ..
            } => self.embedding_contracts.push(EmbeddingContractRecord {
                timestamp_ns: *timestamp_ns,
                name: name.clone(),
                variables: variables.clone(),
            }),
            RunLogEvent::SimSnapshot { .. } => {
                self.sim_snapshots += 1;
            }
            RunLogEvent::BridgeRequest {
                step,
                timestamp_ns,
                request_id,
                method,
                payload_hash,
                ..
            } => self.bridge_records.push(BridgeRecord {
                step: *step,
                timestamp_ns: *timestamp_ns,
                request_id: request_id.clone(),
                method: method.clone(),
                payload_hash: Some(payload_hash.clone()),
                ok: None,
            }),
            RunLogEvent::BridgeResponse {
                step,
                timestamp_ns,
                request_id,
                ok,
                ..
            } => self.bridge_records.push(BridgeRecord {
                step: *step,
                timestamp_ns: *timestamp_ns,
                request_id: request_id.clone(),
                method: "response".to_string(),
                payload_hash: None,
                ok: Some(*ok),
            }),
            RunLogEvent::ActionApplied {
                step,
                timestamp_ns,
                actor,
                action_type,
                payload_hash,
                ..
            } => self.actions.push(ActionRecord {
                step: *step,
                timestamp_ns: *timestamp_ns,
                actor: actor.clone(),
                action_type: action_type.clone(),
                payload_hash: payload_hash.clone(),
            }),
            RunLogEvent::ObjectPose {
                step,
                timestamp_ns,
                object_id,
                pose,
            } => {
                self.object_poses.insert(
                    object_id.clone(),
                    PoseRecord {
                        step: *step,
                        timestamp_ns: *timestamp_ns,
                        pose: pose.clone(),
                    },
                );
            }
            RunLogEvent::FlowGt { .. } => {
                self.flow_gt_records += 1;
            }
            RunLogEvent::FlowPred { .. } => {
                self.flow_pred_records += 1;
            }
            RunLogEvent::PidMetric {
                step,
                timestamp_ns,
                name,
                value,
                ..
            } => {
                self.pid_metric_events += 1;
                self.pid_metrics.insert(
                    name.clone(),
                    MetricRecord {
                        step: *step,
                        timestamp_ns: *timestamp_ns,
                        value: *value,
                    },
                );
            }
            RunLogEvent::GeometryMetric {
                step,
                timestamp_ns,
                name,
                value,
                ..
            } => {
                self.geometry_metric_events += 1;
                self.geometry_metrics.insert(
                    name.clone(),
                    MetricRecord {
                        step: *step,
                        timestamp_ns: *timestamp_ns,
                        value: *value,
                    },
                );
            }
            RunLogEvent::EvaluationMetric {
                step,
                timestamp_ns,
                name,
                value,
                ..
            } => {
                self.evaluation_metric_events += 1;
                self.evaluation_metrics.insert(
                    name.clone(),
                    MetricRecord {
                        step: *step,
                        timestamp_ns: *timestamp_ns,
                        value: *value,
                    },
                );
            }
            RunLogEvent::LabelObserved {
                step,
                timestamp_ns,
                name,
                value,
                ..
            } => self.labels.push(LabelRecord {
                step: *step,
                timestamp_ns: *timestamp_ns,
                name: name.clone(),
                value: value.clone(),
            }),
            RunLogEvent::InterventionApplied {
                step,
                timestamp_ns,
                actor,
                intervention_type,
                payload_hash,
                ..
            } => self.interventions.push(InterventionRecord {
                step: *step,
                timestamp_ns: *timestamp_ns,
                actor: actor.clone(),
                intervention_type: intervention_type.clone(),
                payload_hash: payload_hash.clone(),
            }),
            RunLogEvent::ArtifactLogged {
                timestamp_ns,
                name,
                kind,
                uri,
                sha256,
                ..
            } => self.artifacts.push(ArtifactRecord {
                timestamp_ns: *timestamp_ns,
                name: name.clone(),
                kind: kind.clone(),
                uri: uri.clone(),
                sha256: sha256.clone(),
            }),
            RunLogEvent::AttributionLogged {
                timestamp_ns,
                method,
                target_output,
                layer,
                modality,
                baseline,
                score_hash,
                faithfulness_check,
                artifact_uri,
                ..
            } => self.attributions.push(AttributionRecord {
                timestamp_ns: *timestamp_ns,
                method: method.clone(),
                target_output: target_output.clone(),
                layer: layer.clone(),
                modality: modality.clone(),
                baseline: baseline.clone(),
                score_hash: score_hash.clone(),
                faithfulness_check: *faithfulness_check,
                artifact_uri: artifact_uri.clone(),
            }),
            RunLogEvent::ErrorLogged { message, .. } => self.errors.push(message.clone()),
        }
    }
}

impl RunLogEvent {
    pub fn timestamp_ns(&self) -> u64 {
        match self {
            RunLogEvent::RunStarted { timestamp_ns, .. }
            | RunLogEvent::RunEnded { timestamp_ns, .. }
            | RunLogEvent::ConfigLogged { timestamp_ns, .. }
            | RunLogEvent::FrameObserved { timestamp_ns, .. }
            | RunLogEvent::EmbeddingCaptured { timestamp_ns, .. }
            | RunLogEvent::EmbeddingContract { timestamp_ns, .. }
            | RunLogEvent::SimSnapshot { timestamp_ns, .. }
            | RunLogEvent::BridgeRequest { timestamp_ns, .. }
            | RunLogEvent::BridgeResponse { timestamp_ns, .. }
            | RunLogEvent::ActionApplied { timestamp_ns, .. }
            | RunLogEvent::ObjectPose { timestamp_ns, .. }
            | RunLogEvent::FlowGt { timestamp_ns, .. }
            | RunLogEvent::FlowPred { timestamp_ns, .. }
            | RunLogEvent::PidMetric { timestamp_ns, .. }
            | RunLogEvent::GeometryMetric { timestamp_ns, .. }
            | RunLogEvent::EvaluationMetric { timestamp_ns, .. }
            | RunLogEvent::LabelObserved { timestamp_ns, .. }
            | RunLogEvent::InterventionApplied { timestamp_ns, .. }
            | RunLogEvent::ArtifactLogged { timestamp_ns, .. }
            | RunLogEvent::AttributionLogged { timestamp_ns, .. }
            | RunLogEvent::ErrorLogged { timestamp_ns, .. } => *timestamp_ns,
        }
    }

    pub fn step(&self) -> Option<u64> {
        match self {
            RunLogEvent::FrameObserved { step, .. }
            | RunLogEvent::EmbeddingCaptured { step, .. }
            | RunLogEvent::SimSnapshot { step, .. }
            | RunLogEvent::ActionApplied { step, .. }
            | RunLogEvent::ObjectPose { step, .. }
            | RunLogEvent::FlowGt { step, .. }
            | RunLogEvent::FlowPred { step, .. }
            | RunLogEvent::PidMetric { step, .. }
            | RunLogEvent::GeometryMetric { step, .. }
            | RunLogEvent::EvaluationMetric { step, .. }
            | RunLogEvent::LabelObserved { step, .. }
            | RunLogEvent::InterventionApplied { step, .. } => Some(*step),
            RunLogEvent::BridgeRequest { step, .. }
            | RunLogEvent::BridgeResponse { step, .. }
            | RunLogEvent::ErrorLogged { step, .. } => *step,
            RunLogEvent::RunStarted { .. }
            | RunLogEvent::RunEnded { .. }
            | RunLogEvent::ConfigLogged { .. }
            | RunLogEvent::EmbeddingContract { .. }
            | RunLogEvent::ArtifactLogged { .. }
            | RunLogEvent::AttributionLogged { .. } => None,
        }
    }
}

pub fn runlog_event_contracts() -> Vec<RunLogEventContract> {
    RUN_LOG_EVENT_TYPES
        .iter()
        .map(|event_type| RunLogEventContract {
            event_type: (*event_type).to_string(),
            has_step: matches!(
                *event_type,
                "frame_observed"
                    | "embedding_captured"
                    | "sim_snapshot"
                    | "bridge_request"
                    | "bridge_response"
                    | "action_applied"
                    | "object_pose"
                    | "flow_gt"
                    | "flow_pred"
                    | "pid_metric"
                    | "geometry_metric"
                    | "evaluation_metric"
                    | "label_observed"
                    | "intervention_applied"
                    | "error_logged"
            ),
            carries_payload_hash: matches!(
                *event_type,
                "bridge_request" | "action_applied" | "intervention_applied"
            ),
        })
        .collect()
}

pub fn runlog_contract() -> RunLogContract {
    RunLogContract {
        schema_version: RUN_LOG_SCHEMA_VERSION,
        event_types: runlog_event_contracts(),
        actor_types: ["human_gui", "script", "llm_tool", "system"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        run_statuses: ["succeeded", "failed", "aborted"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        sidecars: RUN_LOG_SIDECARS
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        validation_rules: RUN_LOG_VALIDATION_RULES
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub event_index: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub events: usize,
    pub errors: usize,
    pub warnings: usize,
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors == 0
    }

    fn error(&mut self, event_index: Option<usize>, message: impl Into<String>) {
        self.errors += 1;
        self.issues.push(ValidationIssue {
            severity: ValidationSeverity::Error,
            event_index,
            message: message.into(),
        });
    }

    fn warning(&mut self, event_index: Option<usize>, message: impl Into<String>) {
        self.warnings += 1;
        self.issues.push(ValidationIssue {
            severity: ValidationSeverity::Warning,
            event_index,
            message: message.into(),
        });
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunLogSummary {
    pub schema_version: Option<u32>,
    pub run_id: Option<String>,
    pub config_hash: Option<String>,
    pub status: Option<RunStatus>,
    pub event_count: usize,
    pub trace_hash: String,
    pub validation_errors: usize,
    pub validation_warnings: usize,
    pub last_step: Option<u64>,
    pub last_timestamp_ns: Option<u64>,
    pub actions: usize,
    pub interventions: usize,
    pub objects: usize,
    pub pid_metrics: usize,
    pub geometry_metrics: usize,
    pub evaluation_metrics: usize,
    #[serde(default)]
    pub pid_metric_events: usize,
    #[serde(default)]
    pub geometry_metric_events: usize,
    #[serde(default)]
    pub evaluation_metric_events: usize,
    pub labels: usize,
    pub embeddings: usize,
    pub embedding_contracts: usize,
    pub bridge_records: usize,
    pub sim_snapshots: usize,
    pub artifacts: usize,
    pub attributions: usize,
    pub errors: usize,
    pub flow_gt_records: usize,
    pub flow_pred_records: usize,
    pub validation_issues: Vec<ValidationIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifestEntry {
    pub name: String,
    pub kind: String,
    pub uri: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: u32,
    pub run_id: Option<String>,
    pub config_hash: Option<String>,
    pub run_log_uri: String,
    pub run_log_sha256: Option<String>,
    pub trace_hash: String,
    pub event_count: usize,
    pub validation_errors: usize,
    pub validation_warnings: usize,
    pub artifacts: Vec<ArtifactManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunLogSidecarPaths {
    pub validation: PathBuf,
    pub summary: PathBuf,
    pub manifest: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunLogSidecars {
    pub validation: ValidationReport,
    pub summary: RunLogSummary,
    pub manifest: RunManifest,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarVerificationReport {
    pub checked: usize,
    pub issues: Vec<SidecarVerificationIssue>,
}

impl SidecarVerificationReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }

    fn issue(
        &mut self,
        sidecar: impl Into<String>,
        path: impl AsRef<Path>,
        message: impl Into<String>,
    ) {
        self.issues.push(SidecarVerificationIssue {
            sidecar: sidecar.into(),
            path: path.as_ref().display().to_string(),
            message: message.into(),
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarVerificationIssue {
    pub sidecar: String,
    pub path: String,
    pub message: String,
}

pub fn validate_events(events: &[RunLogEvent]) -> ValidationReport {
    let mut report = ValidationReport {
        events: events.len(),
        ..ValidationReport::default()
    };
    if events.is_empty() {
        report.error(None, "run log is empty");
        return report;
    }

    let mut run_id: Option<&str> = None;
    let mut starts = 0usize;
    let mut ends = 0usize;
    let mut last_timestamp = None;
    let mut last_step = None;
    let mut run_started_config_hash: Option<(usize, String)> = None;
    let mut config_logged_hashes: Vec<(usize, String)> = Vec::new();
    let mut bridge_requests = BTreeSet::new();
    let mut bridge_responses = BTreeSet::new();

    for (idx, event) in events.iter().enumerate() {
        let timestamp = event.timestamp_ns();
        if let Some(prev) = last_timestamp {
            if timestamp < prev {
                report.error(Some(idx), "timestamps must be nondecreasing");
            }
        }
        last_timestamp = Some(timestamp);

        if let Some(step) = event.step() {
            if let Some(prev) = last_step {
                if step < prev {
                    report.error(Some(idx), "steps must be nondecreasing");
                }
            }
            last_step = Some(step);
        }

        match event {
            RunLogEvent::RunStarted {
                schema_version,
                run_id: id,
                config_hash,
                ..
            } => {
                starts += 1;
                if idx != 0 {
                    report.error(Some(idx), "run_started must be the first event");
                }
                if *schema_version != RUN_LOG_SCHEMA_VERSION {
                    report.error(Some(idx), "unsupported run-log schema version");
                }
                if id.is_empty() {
                    report.error(Some(idx), "run_id must not be empty");
                }
                if config_hash.is_empty() {
                    report.warning(Some(idx), "config_hash is empty");
                } else {
                    run_started_config_hash = Some((idx, config_hash.clone()));
                }
                run_id = Some(id);
            }
            RunLogEvent::RunEnded { run_id: id, .. } => {
                ends += 1;
                if idx + 1 != events.len() {
                    report.error(Some(idx), "run_ended must be the last event");
                }
                if let Some(start_id) = run_id {
                    if start_id != id {
                        report.error(Some(idx), "run_ended run_id does not match run_started");
                    }
                }
            }
            RunLogEvent::ActionApplied {
                payload_hash,
                payload,
                action_type,
                ..
            } => {
                validate_payload_hash(&mut report, idx, payload_hash, payload);
                if action_type.is_empty() {
                    report.error(Some(idx), "action_type must not be empty");
                }
            }
            RunLogEvent::InterventionApplied {
                payload_hash,
                payload,
                intervention_type,
                ..
            } => {
                validate_payload_hash(&mut report, idx, payload_hash, payload);
                if intervention_type.is_empty() {
                    report.error(Some(idx), "intervention_type must not be empty");
                }
            }
            RunLogEvent::BridgeRequest {
                request_id,
                method,
                payload_hash,
                payload,
                ..
            } => {
                validate_payload_hash(&mut report, idx, payload_hash, payload);
                if request_id.is_empty() {
                    report.error(Some(idx), "bridge request_id must not be empty");
                } else if !bridge_requests.insert(request_id.clone()) {
                    report.error(Some(idx), "duplicate bridge request_id");
                }
                if method.is_empty() {
                    report.error(Some(idx), "bridge method must not be empty");
                }
            }
            RunLogEvent::BridgeResponse { request_id, .. } => {
                if request_id.is_empty() {
                    report.error(Some(idx), "bridge response request_id must not be empty");
                } else {
                    // Requests are inserted in stream order, so a response whose request_id is
                    // not yet present arrived before (or without) its request — a causality
                    // violation the end-of-stream set difference cannot catch on its own.
                    if !bridge_requests.contains(request_id) {
                        report.error(
                            Some(idx),
                            "bridge response precedes or lacks its matching request",
                        );
                    }
                    if !bridge_responses.insert(request_id.clone()) {
                        report.error(Some(idx), "duplicate bridge response request_id");
                    }
                }
            }
            RunLogEvent::ObjectPose {
                object_id, pose, ..
            } => {
                if object_id.is_empty() {
                    report.error(Some(idx), "object_id must not be empty");
                }
                validate_pose(&mut report, idx, pose);
            }
            RunLogEvent::SimSnapshot { objects, .. } => {
                for object in objects {
                    if object.object_id.is_empty() {
                        report.error(Some(idx), "snapshot object_id must not be empty");
                    }
                    validate_pose(&mut report, idx, &object.pose);
                    validate_vec3(&mut report, idx, object.velocity, "snapshot velocity");
                }
            }
            RunLogEvent::FlowGt {
                object_id, flow, ..
            } => {
                if object_id.is_empty() {
                    report.error(Some(idx), "flow object_id must not be empty");
                }
                for vec in flow {
                    validate_vec3(&mut report, idx, *vec, "flow vector");
                }
            }
            RunLogEvent::FlowPred {
                source,
                object_id,
                horizon_steps,
                flow,
                ..
            } => {
                if source.is_empty() {
                    report.error(Some(idx), "flow source must not be empty");
                }
                if object_id.is_empty() {
                    report.error(Some(idx), "flow object_id must not be empty");
                }
                if *horizon_steps == 0 {
                    report.error(Some(idx), "flow horizon_steps must be positive");
                }
                for vec in flow {
                    validate_vec3(&mut report, idx, *vec, "flow vector");
                }
            }
            RunLogEvent::PidMetric { name, value, .. }
            | RunLogEvent::GeometryMetric { name, value, .. }
            | RunLogEvent::EvaluationMetric { name, value, .. } => {
                if name.is_empty() {
                    report.error(Some(idx), "metric name must not be empty");
                }
                if !value.is_finite() {
                    report.error(Some(idx), "metric value must be finite");
                }
            }
            RunLogEvent::LabelObserved { name, value, .. } => {
                if name.is_empty() {
                    report.error(Some(idx), "label name must not be empty");
                }
                if value.is_null() {
                    report.error(Some(idx), "label value must not be null");
                }
            }
            RunLogEvent::EmbeddingCaptured { name, dims, .. } => {
                if name.is_empty() {
                    report.error(Some(idx), "embedding name must not be empty");
                }
                if dims.is_empty() || dims.contains(&0) {
                    report.error(Some(idx), "embedding dims must be nonempty and positive");
                }
            }
            RunLogEvent::EmbeddingContract {
                name, variables, ..
            } => {
                validate_embedding_contract(&mut report, idx, name, variables);
            }
            RunLogEvent::ArtifactLogged { name, uri, .. } => {
                if name.is_empty() {
                    report.error(Some(idx), "artifact name must not be empty");
                }
                if uri.is_empty() {
                    report.error(Some(idx), "artifact uri must not be empty");
                }
            }
            RunLogEvent::AttributionLogged {
                method,
                target_output,
                ..
            } => {
                if method.is_empty() {
                    report.error(Some(idx), "attribution method must not be empty");
                }
                if target_output.is_empty() {
                    report.error(Some(idx), "attribution target_output must not be empty");
                }
            }
            RunLogEvent::ConfigLogged {
                config_hash,
                config,
                ..
            } => {
                if config_hash.is_empty() {
                    report.warning(Some(idx), "config_hash is empty");
                } else {
                    validate_config_hash(&mut report, idx, config_hash, config);
                    config_logged_hashes.push((idx, config_hash.clone()));
                }
            }
            RunLogEvent::FrameObserved { .. } | RunLogEvent::ErrorLogged { .. } => {}
        }
    }

    if starts != 1 {
        report.error(
            None,
            format!("expected exactly one run_started event, got {starts}"),
        );
    }
    if ends != 1 {
        report.error(
            None,
            format!("expected exactly one run_ended event, got {ends}"),
        );
    }
    for request_id in bridge_responses.difference(&bridge_requests) {
        report.error(
            None,
            format!("bridge response without request: {request_id}"),
        );
    }
    for request_id in bridge_requests.difference(&bridge_responses) {
        report.warning(
            None,
            format!("bridge request without response: {request_id}"),
        );
    }
    if let Some((_, started_hash)) = &run_started_config_hash {
        for (idx, logged_hash) in &config_logged_hashes {
            if logged_hash != started_hash {
                report.error(
                    Some(*idx),
                    "config_logged config_hash does not match run_started",
                );
            }
        }
    } else if !config_logged_hashes.is_empty() {
        // run_started.config_hash was empty (only a warning above), so there is no anchor to
        // cross-check the config_logged events against — their integrity cannot be verified.
        report.error(
            None,
            "run_started.config_hash is empty but config_logged events are present; config integrity cannot be verified",
        );
    }

    report
}

pub fn validate_events_from_path(path: impl AsRef<Path>) -> Result<ValidationReport> {
    Ok(validate_events(&read_events_from_path(path)?))
}

pub fn summarize_events(events: &[RunLogEvent]) -> Result<RunLogSummary> {
    let state = replay_events(events);
    let validation = validate_events(events);
    let trace_hash = replay_trace_hash(events)?;
    Ok(RunLogSummary {
        schema_version: state.schema_version,
        run_id: state.run_id,
        config_hash: state.config_hash,
        status: state.status,
        event_count: state.events_seen,
        trace_hash,
        validation_errors: validation.errors,
        validation_warnings: validation.warnings,
        last_step: state.last_step,
        last_timestamp_ns: state.last_timestamp_ns,
        actions: state.actions.len(),
        interventions: state.interventions.len(),
        objects: state.object_poses.len(),
        pid_metrics: state.pid_metrics.len(),
        geometry_metrics: state.geometry_metrics.len(),
        evaluation_metrics: state.evaluation_metrics.len(),
        pid_metric_events: state.pid_metric_events,
        geometry_metric_events: state.geometry_metric_events,
        evaluation_metric_events: state.evaluation_metric_events,
        labels: state.labels.len(),
        embeddings: state.embeddings.len(),
        embedding_contracts: state.embedding_contracts.len(),
        bridge_records: state.bridge_records.len(),
        sim_snapshots: state.sim_snapshots,
        artifacts: state.artifacts.len(),
        attributions: state.attributions.len(),
        errors: state.errors.len(),
        flow_gt_records: state.flow_gt_records,
        flow_pred_records: state.flow_pred_records,
        validation_issues: validation.issues,
    })
}

pub fn summarize_path(path: impl AsRef<Path>) -> Result<RunLogSummary> {
    summarize_events(&read_events_from_path(path)?)
}

pub fn manifest_for_events(path: impl AsRef<Path>, events: &[RunLogEvent]) -> Result<RunManifest> {
    let path = path.as_ref();
    let summary = summarize_events(events)?;
    let state = replay_events(events);
    Ok(RunManifest {
        schema_version: RUN_LOG_SCHEMA_VERSION,
        run_id: summary.run_id,
        config_hash: summary.config_hash,
        run_log_uri: path.display().to_string(),
        run_log_sha256: Some(sha256_file(path)?),
        trace_hash: summary.trace_hash,
        event_count: summary.event_count,
        validation_errors: summary.validation_errors,
        validation_warnings: summary.validation_warnings,
        artifacts: state
            .artifacts
            .into_iter()
            .map(|artifact| ArtifactManifestEntry {
                name: artifact.name,
                kind: artifact.kind,
                uri: artifact.uri,
                sha256: artifact.sha256,
            })
            .collect(),
    })
}

pub fn manifest_for_path(path: impl AsRef<Path>) -> Result<RunManifest> {
    let path = path.as_ref();
    manifest_for_events(path, &read_events_from_path(path)?)
}

pub fn runlog_sidecar_paths(path: impl AsRef<Path>) -> RunLogSidecarPaths {
    let path = path.as_ref();
    RunLogSidecarPaths {
        validation: sidecar_path(path, "validation"),
        summary: sidecar_path(path, "summary"),
        manifest: sidecar_path(path, "manifest"),
    }
}

pub fn sidecars_for_path(path: impl AsRef<Path>) -> Result<RunLogSidecars> {
    let path = path.as_ref();
    let events = read_events_from_path(path)?;
    Ok(RunLogSidecars {
        validation: validate_events(&events),
        summary: summarize_events(&events)?,
        manifest: manifest_for_events(path, &events)?,
    })
}

pub fn write_sidecars_for_path(path: impl AsRef<Path>) -> Result<RunLogSidecarPaths> {
    let path = path.as_ref();
    let paths = runlog_sidecar_paths(path);
    let sidecars = sidecars_for_path(path)?;
    write_json_file(&paths.validation, &sidecars.validation)?;
    write_json_file(&paths.summary, &sidecars.summary)?;
    write_json_file(&paths.manifest, &sidecars.manifest)?;
    Ok(paths)
}

pub fn verify_sidecars_for_path(path: impl AsRef<Path>) -> Result<SidecarVerificationReport> {
    let path = path.as_ref();
    let paths = runlog_sidecar_paths(path);
    let expected = sidecars_for_path(path)?;
    let mut report = SidecarVerificationReport::default();
    verify_sidecar(
        &mut report,
        "validation",
        &paths.validation,
        &expected.validation,
    );
    verify_sidecar(&mut report, "summary", &paths.summary, &expected.summary);
    verify_sidecar(&mut report, "manifest", &paths.manifest, &expected.manifest);
    Ok(report)
}

fn verify_sidecar<T>(
    report: &mut SidecarVerificationReport,
    sidecar: &str,
    path: impl AsRef<Path>,
    expected: &T,
) where
    T: Serialize,
{
    let path = path.as_ref();
    report.checked += 1;
    let file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == ErrorKind::NotFound => {
            report.issue(sidecar, path, "sidecar is missing");
            return;
        }
        Err(err) => {
            report.issue(sidecar, path, format!("failed to open sidecar: {err}"));
            return;
        }
    };
    let actual = match serde_json::from_reader::<_, serde_json::Value>(file) {
        Ok(actual) => actual,
        Err(err) => {
            report.issue(sidecar, path, format!("invalid sidecar JSON: {err}"));
            return;
        }
    };
    let expected = match serde_json::to_value(expected) {
        Ok(expected) => expected,
        Err(err) => {
            report.issue(
                sidecar,
                path,
                format!("failed to serialize expected sidecar: {err}"),
            );
            return;
        }
    };
    if actual != expected {
        report.issue(sidecar, path, "sidecar does not match current run log");
    }
}

pub fn write_json_file<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let file = File::create(path.as_ref())
        .with_context(|| format!("failed to create {}", path.as_ref().display()))?;
    serde_json::to_writer_pretty(file, value)
        .with_context(|| format!("failed to write {}", path.as_ref().display()))
}

fn validate_payload_hash(
    report: &mut ValidationReport,
    event_index: usize,
    payload_hash: &str,
    payload: &serde_json::Value,
) {
    match canonical_json_hash(payload) {
        Ok(expected) if expected == payload_hash => {}
        Ok(_) => report.error(Some(event_index), "payload_hash does not match payload"),
        Err(err) => report.error(Some(event_index), format!("payload hash failed: {err}")),
    }
}

fn validate_config_hash(
    report: &mut ValidationReport,
    event_index: usize,
    config_hash: &str,
    config: &serde_json::Value,
) {
    match canonical_json_hash(config) {
        Ok(expected) if expected == config_hash => {}
        Ok(_) => report.error(Some(event_index), "config_hash does not match config"),
        Err(err) => report.error(Some(event_index), format!("config hash failed: {err}")),
    }
}

fn validate_pose(report: &mut ValidationReport, event_index: usize, pose: &Pose) {
    validate_vec3(report, event_index, pose.position, "pose position");
    for value in pose.orientation_xyzw {
        if !value.is_finite() {
            report.error(Some(event_index), "pose orientation must be finite");
        }
    }
}

fn validate_embedding_contract(
    report: &mut ValidationReport,
    event_index: usize,
    name: &str,
    variables: &[EmbeddingVariableContract],
) {
    if name.is_empty() {
        report.error(
            Some(event_index),
            "embedding contract name must not be empty",
        );
    }
    if variables.is_empty() {
        report.error(
            Some(event_index),
            "embedding contract must include at least one variable",
        );
    }
    let mut seen = BTreeSet::new();
    for variable in variables {
        if variable.variable.is_empty() {
            report.error(
                Some(event_index),
                "embedding contract variable name must not be empty",
            );
        } else if !seen.insert(variable.variable.clone()) {
            report.error(
                Some(event_index),
                format!(
                    "duplicate embedding contract variable {}",
                    variable.variable
                ),
            );
        }
        if variable.source.is_empty() {
            report.error(
                Some(event_index),
                "embedding contract source must not be empty",
            );
        }
        if variable.dims.is_empty() || variable.dims.contains(&0) {
            report.error(
                Some(event_index),
                "embedding contract dims must be nonempty and positive",
            );
        }
    }
}

fn validate_vec3(report: &mut ValidationReport, event_index: usize, value: [f64; 3], field: &str) {
    if value.iter().any(|v| !v.is_finite()) {
        report.error(Some(event_index), format!("{field} must be finite"));
    }
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "runlog".into());
    path.with_file_name(format!("{file_name}.{suffix}.json"))
}

pub struct RunLogWriter<W> {
    writer: W,
}

impl RunLogWriter<BufWriter<File>> {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("failed to create run log {}", path.as_ref().display()))?;
        Ok(Self::new(BufWriter::new(file)))
    }
}

impl<W: Write> RunLogWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn append(&mut self, event: &RunLogEvent) -> Result<()> {
        serde_json::to_writer(&mut self.writer, event)
            .context("failed to serialize run-log event")?;
        self.writer
            .write_all(b"\n")
            .context("failed to write run-log newline")?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("failed to flush run log")
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

pub fn read_events_from_path(path: impl AsRef<Path>) -> Result<Vec<RunLogEvent>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("failed to open run log {}", path.as_ref().display()))?;
    read_events(BufReader::new(file))
}

pub fn replay_state_from_path(path: impl AsRef<Path>) -> Result<ReplayState> {
    Ok(replay_events(&read_events_from_path(path)?))
}

pub fn replay_trace_hash_from_path(path: impl AsRef<Path>) -> Result<String> {
    replay_trace_hash(&read_events_from_path(path)?)
}

pub fn read_events<R: BufRead>(reader: R) -> Result<Vec<RunLogEvent>> {
    let mut events = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read run-log line {}", idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str(&line)
            .with_context(|| format!("invalid run-log event at line {}", idx + 1))?;
        events.push(event);
    }
    Ok(events)
}

pub fn replay_events(events: &[RunLogEvent]) -> ReplayState {
    let mut state = ReplayState::default();
    for event in events {
        state.apply(event);
    }
    state
}

/// Order-sensitive content hash over the **full** event sequence.
///
/// This digests every event in order, so it detects *any* change to the trace — a reordered
/// event, or a changed intermediate metric/pose/observation value. A hash of the collapsed
/// [`ReplayState`] cannot: that state is last-wins for metrics/poses and drops per-frame
/// observation hashes, so two materially different logs that happen to reach the same final
/// state would collide (and a `--compare` would falsely report a match). Each event's canonical
/// JSON is length-prefixed before folding into the SHA-256 so record boundaries are unambiguous.
pub fn replay_trace_hash(events: &[RunLogEvent]) -> Result<String> {
    let mut hasher = Sha256::new();
    for event in events {
        let bytes =
            serde_json::to_vec(event).context("failed to serialize event for trace hash")?;
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(to_hex(&hasher.finalize()))
}

pub fn canonical_json_hash<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).context("failed to serialize value for hashing")?;
    Ok(sha256_hex(&bytes))
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path.as_ref())
        .with_context(|| format!("failed to open artifact {}", path.as_ref().display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("failed to read artifact {}", path.as_ref().display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(to_hex(&hasher.finalize()))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finalize())
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Cursor;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn actor() -> Actor {
        Actor {
            actor_type: ActorType::Script,
            actor_id: "test".to_string(),
            session_id: Some("s1".to_string()),
        }
    }

    fn sample_events() -> Vec<RunLogEvent> {
        let step_payload = json!({ "dt": 0.01 });
        let step_payload_hash = canonical_json_hash(&step_payload).unwrap();
        vec![
            RunLogEvent::RunStarted {
                schema_version: RUN_LOG_SCHEMA_VERSION,
                run_id: "run-1".to_string(),
                timestamp_ns: 1,
                config_hash: "cfg".to_string(),
                metadata: BTreeMap::new(),
            },
            RunLogEvent::ActionApplied {
                step: 0,
                timestamp_ns: 2,
                actor: actor(),
                action_type: "sim.step".to_string(),
                payload_hash: step_payload_hash.clone(),
                payload: step_payload.clone(),
            },
            RunLogEvent::EmbeddingCaptured {
                step: 0,
                timestamp_ns: 2,
                name: "V".to_string(),
                dims: vec![1, 3],
                artifact_uri: Some("artifacts/v.npy".to_string()),
                sha256: Some("abc".to_string()),
                metadata: BTreeMap::new(),
            },
            RunLogEvent::EmbeddingContract {
                timestamp_ns: 2,
                name: "vla_tuple".to_string(),
                variables: vec![EmbeddingVariableContract {
                    variable: "V".to_string(),
                    source: "V".to_string(),
                    dims: vec![1, 3],
                    artifact_uri: Some("artifacts/v.npy".to_string()),
                    sha256: Some("abc".to_string()),
                }],
                metadata: BTreeMap::new(),
            },
            RunLogEvent::BridgeRequest {
                step: Some(0),
                timestamp_ns: 2,
                request_id: "req-1".to_string(),
                actor: actor(),
                method: "sim.step".to_string(),
                payload_hash: step_payload_hash,
                payload: step_payload,
            },
            RunLogEvent::BridgeResponse {
                step: Some(0),
                timestamp_ns: 2,
                request_id: "req-1".to_string(),
                ok: true,
                message: None,
                result_hash: None,
            },
            RunLogEvent::ObjectPose {
                step: 0,
                timestamp_ns: 3,
                object_id: "cube".to_string(),
                pose: Pose {
                    position: [1.0, 2.0, 3.0],
                    orientation_xyzw: [0.0, 0.0, 0.0, 1.0],
                },
            },
            RunLogEvent::PidMetric {
                step: 0,
                timestamp_ns: 4,
                name: "redundancy".to_string(),
                value: 0.25,
                metadata: BTreeMap::new(),
            },
            RunLogEvent::EvaluationMetric {
                step: 0,
                timestamp_ns: 4,
                name: "baseline.accuracy".to_string(),
                value: 0.75,
                metadata: BTreeMap::new(),
            },
            RunLogEvent::LabelObserved {
                step: 0,
                timestamp_ns: 4,
                name: "success".to_string(),
                value: json!(true),
                metadata: BTreeMap::new(),
            },
            RunLogEvent::RunEnded {
                run_id: "run-1".to_string(),
                timestamp_ns: 5,
                status: RunStatus::Succeeded,
                message: None,
            },
        ]
    }

    #[test]
    fn jsonl_round_trip_preserves_events() {
        let events = sample_events();
        let mut writer = RunLogWriter::new(Vec::new());
        for event in &events {
            writer.append(event).unwrap();
        }
        let bytes = writer.into_inner();
        let decoded = read_events(Cursor::new(bytes)).unwrap();
        assert_eq!(decoded, events);
    }

    #[test]
    fn replay_tracks_latest_state() {
        let events = sample_events();
        let state = replay_events(&events);
        assert_eq!(state.run_id.as_deref(), Some("run-1"));
        assert_eq!(state.last_step, Some(0));
        assert_eq!(state.actions.len(), 1);
        assert_eq!(state.embeddings.len(), 1);
        assert_eq!(state.embedding_contracts[0].name, "vla_tuple");
        assert_eq!(state.embedding_contracts[0].variables[0].variable, "V");
        assert_eq!(state.bridge_records.len(), 2);
        assert_eq!(state.object_poses["cube"].pose.position, [1.0, 2.0, 3.0]);
        assert_eq!(state.pid_metrics["redundancy"].value, 0.25);
        assert_eq!(state.evaluation_metrics["baseline.accuracy"].value, 0.75);
        assert_eq!(state.pid_metric_events, 1);
        assert_eq!(state.geometry_metric_events, 0);
        assert_eq!(state.evaluation_metric_events, 1);
        assert_eq!(state.labels[0].name, "success");
        assert_eq!(state.labels[0].value, json!(true));
    }

    #[test]
    fn replay_counts_repeated_metric_events_separately_from_unique_names() {
        let mut events = sample_events();
        events.insert(
            events.len() - 1,
            RunLogEvent::EvaluationMetric {
                step: 0,
                timestamp_ns: 4,
                name: "baseline.accuracy".to_string(),
                value: 0.875,
                metadata: BTreeMap::new(),
            },
        );

        let state = replay_events(&events);
        assert_eq!(state.evaluation_metrics.len(), 1);
        assert_eq!(state.evaluation_metrics["baseline.accuracy"].value, 0.875);
        assert_eq!(state.evaluation_metric_events, 2);

        let summary = summarize_events(&events).unwrap();
        assert_eq!(summary.evaluation_metrics, 1);
        assert_eq!(summary.evaluation_metric_events, 2);
    }

    #[test]
    fn validation_accepts_sample_events() {
        let report = validate_events(&sample_events());
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.warnings, 0);
    }

    #[test]
    fn validation_catches_bad_payload_hash() {
        let mut events = sample_events();
        if let RunLogEvent::ActionApplied { payload_hash, .. } = &mut events[1] {
            *payload_hash = "bad".to_string();
        }
        let report = validate_events(&events);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("payload_hash")));
    }

    #[test]
    fn validation_catches_bad_config_hash() {
        let config = json!({ "dt": 0.1 });
        let events = vec![
            RunLogEvent::RunStarted {
                schema_version: RUN_LOG_SCHEMA_VERSION,
                run_id: "run-1".to_string(),
                timestamp_ns: 1,
                config_hash: canonical_json_hash(&config).unwrap(),
                metadata: BTreeMap::new(),
            },
            RunLogEvent::ConfigLogged {
                timestamp_ns: 1,
                config_hash: "bad".to_string(),
                config,
            },
            RunLogEvent::RunEnded {
                run_id: "run-1".to_string(),
                timestamp_ns: 2,
                status: RunStatus::Failed,
                message: None,
            },
        ];
        let report = validate_events(&events);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("config_hash")));
    }

    #[test]
    fn validation_catches_config_hash_mismatch_with_run_started() {
        let config = json!({ "dt": 0.1 });
        let config_hash = canonical_json_hash(&config).unwrap();
        let events = vec![
            RunLogEvent::RunStarted {
                schema_version: RUN_LOG_SCHEMA_VERSION,
                run_id: "run-1".to_string(),
                timestamp_ns: 1,
                config_hash: "different".to_string(),
                metadata: BTreeMap::new(),
            },
            RunLogEvent::ConfigLogged {
                timestamp_ns: 1,
                config_hash,
                config,
            },
            RunLogEvent::RunEnded {
                run_id: "run-1".to_string(),
                timestamp_ns: 2,
                status: RunStatus::Failed,
                message: None,
            },
        ];
        let report = validate_events(&events);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("does not match run_started")));
    }

    #[test]
    fn validation_catches_events_after_run_ended() {
        let mut events = sample_events();
        events.push(RunLogEvent::ErrorLogged {
            step: None,
            timestamp_ns: 6,
            message: "late event".to_string(),
            recoverable: true,
        });
        let report = validate_events(&events);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("run_ended must be the last event")));
    }

    #[test]
    fn validation_catches_bad_label() {
        let mut events = sample_events();
        if let RunLogEvent::LabelObserved { name, value, .. } = &mut events[9] {
            name.clear();
            *value = serde_json::Value::Null;
        }
        let report = validate_events(&events);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("label name")));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("label value")));
    }

    #[test]
    fn validation_catches_bad_embedding_contract() {
        let mut events = sample_events();
        if let RunLogEvent::EmbeddingContract {
            name, variables, ..
        } = &mut events[3]
        {
            name.clear();
            variables.push(EmbeddingVariableContract {
                variable: "V".to_string(),
                source: "".to_string(),
                dims: vec![0],
                artifact_uri: None,
                sha256: None,
            });
        }
        let report = validate_events(&events);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("embedding contract name")));
        assert!(report.issues.iter().any(|issue| issue
            .message
            .contains("duplicate embedding contract variable")));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("embedding contract dims")));
    }

    #[test]
    fn validation_catches_bad_flow_pred() {
        let mut events = sample_events();
        events.insert(
            events.len() - 1,
            RunLogEvent::FlowPred {
                step: 0,
                timestamp_ns: 4,
                source: "".to_string(),
                object_id: "".to_string(),
                horizon_steps: 0,
                flow: vec![[f64::NAN, 0.0, 0.0]],
                metadata: BTreeMap::new(),
            },
        );
        let report = validate_events(&events);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("flow source")));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("horizon_steps")));
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("flow vector")));
    }

    #[test]
    fn summary_and_manifest_include_trace_hash() {
        let mut events = sample_events();
        events.insert(
            events.len() - 1,
            RunLogEvent::FlowPred {
                step: 0,
                timestamp_ns: 4,
                source: "constant_velocity_baseline".to_string(),
                object_id: "cube".to_string(),
                horizon_steps: 1,
                flow: vec![[0.1, 0.0, 0.0]],
                metadata: BTreeMap::new(),
            },
        );
        let summary = summarize_events(&events).unwrap();
        assert_eq!(summary.run_id.as_deref(), Some("run-1"));
        assert_eq!(summary.config_hash.as_deref(), Some("cfg"));
        assert_eq!(summary.validation_errors, 0);
        assert_eq!(summary.evaluation_metrics, 1);
        assert_eq!(summary.evaluation_metric_events, 1);
        assert_eq!(summary.labels, 1);
        assert_eq!(summary.embedding_contracts, 1);
        assert_eq!(summary.flow_pred_records, 1);
        assert_eq!(summary.trace_hash.len(), 64);
        let state_hash = replay_trace_hash(&events).unwrap();
        assert_eq!(summary.trace_hash, state_hash);
    }

    #[test]
    fn trace_hash_distinguishes_traces_with_identical_final_state() {
        // `FrameObserved` is a no-op for `ReplayState`, so two traces differing ONLY in a
        // frame's `observation_hash` collapse to the SAME final state — yet they are different
        // traces. A hash of the collapsed state collides (and `--compare` would falsely report a
        // match); the full event-sequence trace hash must distinguish them.
        let make = |obs: &str| {
            vec![
                RunLogEvent::RunStarted {
                    schema_version: RUN_LOG_SCHEMA_VERSION,
                    run_id: "run-1".to_string(),
                    timestamp_ns: 1,
                    config_hash: "cfg".to_string(),
                    metadata: BTreeMap::new(),
                },
                RunLogEvent::FrameObserved {
                    step: 0,
                    timestamp_ns: 2,
                    observation_hash: Some(obs.to_string()),
                    metadata: BTreeMap::new(),
                },
                RunLogEvent::RunEnded {
                    run_id: "run-1".to_string(),
                    timestamp_ns: 3,
                    status: RunStatus::Succeeded,
                    message: None,
                },
            ]
        };
        let a = make("frame-aaaa");
        let b = make("frame-bbbb");
        // Precondition: the collapsed replay states are byte-identical...
        assert_eq!(
            canonical_json_hash(&replay_events(&a)).unwrap(),
            canonical_json_hash(&replay_events(&b)).unwrap(),
            "the two traces must collapse to the same ReplayState for this test to be meaningful"
        );
        // ...but the full-trace hashes must differ.
        assert_ne!(
            replay_trace_hash(&a).unwrap(),
            replay_trace_hash(&b).unwrap(),
            "trace hash must reflect per-event content, not just the final collapsed state"
        );
    }

    #[test]
    fn replay_trace_hash_is_stable() {
        let events = sample_events();
        let h1 = replay_trace_hash(&events).unwrap();
        let h2 = replay_trace_hash(&events).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn runlog_contract_lists_current_schema_surface() {
        let contract = runlog_contract();
        assert_eq!(contract.schema_version, RUN_LOG_SCHEMA_VERSION);
        assert_eq!(contract.event_types.len(), RUN_LOG_EVENT_TYPES.len());
        assert!(contract
            .event_types
            .iter()
            .any(|event| event.event_type == "bridge_request"
                && event.has_step
                && event.carries_payload_hash));
        assert!(contract
            .event_types
            .iter()
            .any(|event| event.event_type == "evaluation_metric" && event.has_step));
        assert!(contract
            .event_types
            .iter()
            .any(|event| event.event_type == "label_observed" && event.has_step));
        assert!(contract
            .event_types
            .iter()
            .any(|event| event.event_type == "embedding_contract" && !event.has_step));
        assert!(contract
            .event_types
            .iter()
            .any(|event| event.event_type == "flow_pred" && event.has_step));
        assert!(contract.sidecars.contains(&"manifest".to_string()));
        assert!(contract.actor_types.contains(&"llm_tool".to_string()));
    }

    #[test]
    fn malformed_json_reports_line_number() {
        let mut writer = RunLogWriter::new(Vec::new());
        writer.append(&sample_events()[0]).unwrap();
        let mut bytes = writer.into_inner();
        bytes.extend_from_slice(b"not-json\n");
        let err = read_events(Cursor::new(bytes)).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("line 2"));
    }

    #[test]
    fn sidecar_paths_append_suffixes_to_runlog_name() {
        let paths = runlog_sidecar_paths("outputs/demo.jsonl");
        assert_eq!(
            paths.validation,
            PathBuf::from("outputs/demo.jsonl.validation.json")
        );
        assert_eq!(
            paths.summary,
            PathBuf::from("outputs/demo.jsonl.summary.json")
        );
        assert_eq!(
            paths.manifest,
            PathBuf::from("outputs/demo.jsonl.manifest.json")
        );
    }

    #[test]
    fn write_sidecars_emits_validation_summary_and_manifest() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("pid-runlog-sidecars-{stamp}.jsonl"));
        let mut writer = RunLogWriter::create(&path).unwrap();
        for event in sample_events() {
            writer.append(&event).unwrap();
        }
        writer.flush().unwrap();

        let paths = write_sidecars_for_path(&path).unwrap();
        let validation: ValidationReport =
            serde_json::from_reader(File::open(&paths.validation).unwrap()).unwrap();
        let summary: RunLogSummary =
            serde_json::from_reader(File::open(&paths.summary).unwrap()).unwrap();
        let manifest: RunManifest =
            serde_json::from_reader(File::open(&paths.manifest).unwrap()).unwrap();

        assert!(validation.is_valid(), "{:?}", validation.issues);
        assert_eq!(summary.run_id.as_deref(), Some("run-1"));
        assert_eq!(manifest.event_count, summary.event_count);
        assert_eq!(manifest.trace_hash, summary.trace_hash);
        assert_eq!(manifest.config_hash, summary.config_hash);

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(paths.validation);
        let _ = std::fs::remove_file(paths.summary);
        let _ = std::fs::remove_file(paths.manifest);
    }

    #[test]
    fn verify_sidecars_accepts_current_sidecars() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("pid-runlog-verify-sidecars-{stamp}.jsonl"));
        let mut writer = RunLogWriter::create(&path).unwrap();
        for event in sample_events() {
            writer.append(&event).unwrap();
        }
        writer.flush().unwrap();

        let paths = write_sidecars_for_path(&path).unwrap();
        let report = verify_sidecars_for_path(&path).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.checked, 3);

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(paths.validation);
        let _ = std::fs::remove_file(paths.summary);
        let _ = std::fs::remove_file(paths.manifest);
    }

    #[test]
    fn verify_sidecars_reports_extra_summary_fields() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("pid-runlog-stale-sidecar-{stamp}.jsonl"));
        let mut writer = RunLogWriter::create(&path).unwrap();
        for event in sample_events() {
            writer.append(&event).unwrap();
        }
        writer.flush().unwrap();

        let paths = write_sidecars_for_path(&path).unwrap();
        let mut stale_summary: serde_json::Value =
            serde_json::from_reader(File::open(&paths.summary).unwrap()).unwrap();
        stale_summary
            .as_object_mut()
            .unwrap()
            .insert("stale_extra_field".to_string(), true.into());
        write_json_file(&paths.summary, &stale_summary).unwrap();

        let report = verify_sidecars_for_path(&path).unwrap();
        assert!(!report.is_valid());
        assert!(report.issues.iter().any(|issue| {
            issue.sidecar == "summary" && issue.message.contains("does not match")
        }));

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(paths.validation);
        let _ = std::fs::remove_file(paths.summary);
        let _ = std::fs::remove_file(paths.manifest);
    }
}
