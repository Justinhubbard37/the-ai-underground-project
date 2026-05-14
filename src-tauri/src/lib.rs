use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use chrono::Utc;
use reqwest::{Client, Url};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

const LOCAL_GEMMA_TEMPERATURE: f32 = 0.2;
const LOCAL_GEMMA_DEFAULT_TIMEOUT_SECONDS: u64 = 120;
const POLL_INTERVAL: Duration = Duration::from_secs(10);

const VECTOR_SCHEMA_SQL: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS vector_memory USING vec0(
  id TEXT PRIMARY KEY,
  namespace TEXT,
  embedding float[768],
  raw_text TEXT,
  created_at TEXT
);
"#;

#[derive(Debug, Deserialize)]
struct ExecuteLocalGemmaRequest {
    prompt: String,
    scalpel_context: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct RuntimeConfigInput {
    endpoint: String,
    model: String,
    profile_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DispatchDeepResearchRequest {
    query: String,
    depth_mode: DepthMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum DepthMode {
    Standard,
    Max,
}

#[derive(Debug, Serialize)]
struct LocalRunStarted {
    run_id: String,
}

#[derive(Debug, Serialize)]
struct DeepResearchStarted {
    task_id: String,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ActiveModelIdentityStatus {
    Configured,
    MissingEndpoint,
    MissingModel,
    MissingProfile,
    UnknownProfile,
    InvalidEndpoint,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
struct ActiveModelIdentity {
    status: ActiveModelIdentityStatus,
    profile_id: Option<String>,
    display_name: String,
    runtime_label: String,
    runtime_model_id: Option<String>,
    endpoint_configured: bool,
    model_configured: bool,
    profile_known: bool,
    detail: Option<String>,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RuntimeConfigSource {
    Session,
    Environment,
    Missing,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
struct RuntimeConfigStatus {
    source: RuntimeConfigSource,
    valid: bool,
    status: ActiveModelIdentityStatus,
    display_name: String,
    profile_id: Option<String>,
    endpoint_configured: bool,
    model_configured: bool,
    profile_known: bool,
    sanitized_endpoint_label: Option<String>,
    detail: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct LocalGemmaUpdate {
    run_id: String,
    delta: String,
    done: bool,
    error: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct ResearchUpdate {
    task_id: String,
    query: String,
    status: ResearchStatus,
    active_step: String,
    thought_summary: Option<String>,
    steps: Vec<ResearchStep>,
    artifact: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum ResearchStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct ResearchStep {
    id: Option<String>,
    title: Option<String>,
    active_step: Option<String>,
    thought_summary: Option<String>,
    status: Option<ResearchStatus>,
}

#[derive(Debug, Deserialize)]
struct GenericInteractionResponse {
    task_id: Option<String>,
    id: Option<String>,
    status: Option<ResearchStatus>,
    active_step: Option<String>,
    thought_summary: Option<String>,
    steps: Option<Vec<ResearchStep>>,
    artifact: Option<String>,
    output: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct LocalGemmaConfig {
    endpoint: Url,
    model: String,
    profile: Option<ModelCapabilityProfile>,
    timeout: Duration,
}

#[derive(Debug, Default)]
struct RuntimeConfigState {
    session_config: Mutex<Option<AppliedRuntimeConfig>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AppliedRuntimeConfig {
    endpoint: String,
    model: String,
    profile_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EffectiveRuntimeConfigValues {
    source: RuntimeConfigSource,
    endpoint: Option<String>,
    model: Option<String>,
    profile_id: Option<String>,
}

#[derive(Debug, Clone)]
struct ContextPacket {
    prompt: String,
    scalpel_context: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapabilitySupport {
    Yes,
    No,
    Unknown,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeEngine {
    LmStudio,
    LlamaCppSidecar,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ModelCapabilityProfile {
    profile_id: &'static str,
    display_name: &'static str,
    runtime_model_id: String,
    runtime_engine: RuntimeEngine,
    supports_text_input: CapabilitySupport,
    supports_image_input: CapabilitySupport,
    supports_video_frame_input: CapabilitySupport,
    supports_function_calling: CapabilitySupport,
    supports_structured_output: CapabilitySupport,
    supports_streaming: CapabilitySupport,
    supports_embeddings: CapabilitySupport,
    default_enabled_tools: Vec<&'static str>,
    risk_notes: &'static str,
    explicit_uncensored_label: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolImplementationStatus {
    Planned,
    Disabled,
    Mvp,
    Active,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolRiskLevel {
    Low,
    Medium,
    High,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolLocation {
    Local,
    RemoteCapable,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ToolRegistryEntry {
    tool_id: &'static str,
    display_name: &'static str,
    description: &'static str,
    enabled: bool,
    local_or_remote: ToolLocation,
    requires_user_approval: bool,
    input_schema: serde_json::Value,
    output_schema: serde_json::Value,
    supported_model_capabilities: Vec<&'static str>,
    risk_level: ToolRiskLevel,
    ui_affordance_requirement: &'static str,
    implementation_status: ToolImplementationStatus,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum UserApprovalState {
    NotRequired,
    Required,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelResponseEvent {
    ModelText(String),
    ToolCallRequested {
        approval_required: UserApprovalState,
    },
    Error(String),
    Complete,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionRequest {
    model: String,
    messages: Vec<OpenAiChatMessage>,
    temperature: f32,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OpenAiChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatCompletionResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: Option<OpenAiMessage>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
    tool_calls: Option<Vec<serde_json::Value>>,
}

#[tauri::command]
async fn execute_local_gemma(
    app: AppHandle,
    payload: ExecuteLocalGemmaRequest,
) -> Result<LocalRunStarted, String> {
    let prompt = payload.prompt.trim();
    if prompt.is_empty() {
        return Err("Prompt is required.".to_string());
    }

    let run_id = Uuid::new_v4().to_string();
    let task_run_id = run_id.clone();
    let task_payload = ExecuteLocalGemmaRequest {
        prompt: prompt.to_string(),
        scalpel_context: payload
            .scalpel_context
            .as_ref()
            .map(|context| context.trim().to_string())
            .filter(|context| !context.is_empty()),
    };

    tauri::async_runtime::spawn(async move {
        run_local_gemma(app, task_run_id, task_payload).await;
    });

    Ok(LocalRunStarted { run_id })
}

#[tauri::command]
async fn dispatch_deep_research(
    app: AppHandle,
    payload: DispatchDeepResearchRequest,
) -> Result<DeepResearchStarted, String> {
    let query = payload.query.trim();
    if query.is_empty() {
        return Err("Research query is required.".to_string());
    }

    let task_id = Uuid::new_v4().to_string();
    let task_payload = DispatchDeepResearchRequest {
        query: query.to_string(),
        depth_mode: payload.depth_mode,
    };
    let poll_task_id = task_id.clone();

    emit_research_update(
        &app,
        ResearchUpdate {
            task_id: task_id.clone(),
            query: query.to_string(),
            status: ResearchStatus::Queued,
            active_step: "Queued for Interactions API dispatch.".to_string(),
            thought_summary: None,
            steps: vec![],
            artifact: None,
            error: None,
        },
    );

    tauri::async_runtime::spawn(async move {
        run_deep_research_polling(app, poll_task_id, task_payload).await;
    });

    Ok(DeepResearchStarted { task_id })
}

#[tauri::command]
fn get_active_model_identity(runtime_config: State<'_, RuntimeConfigState>) -> ActiveModelIdentity {
    active_model_identity_from_values(effective_runtime_config_values(&runtime_config))
}

#[tauri::command]
fn get_runtime_config_status(runtime_config: State<'_, RuntimeConfigState>) -> RuntimeConfigStatus {
    runtime_config_status_from_values(effective_runtime_config_values(&runtime_config))
}

#[tauri::command]
fn validate_runtime_config(payload: RuntimeConfigInput) -> RuntimeConfigStatus {
    runtime_config_status_from_input(&payload)
}

#[tauri::command]
fn apply_runtime_config(
    runtime_config: State<'_, RuntimeConfigState>,
    payload: RuntimeConfigInput,
) -> RuntimeConfigStatus {
    match runtime_config.session_config.lock() {
        Ok(mut session_config) => {
            let status = apply_runtime_config_to_session(&mut session_config, &payload);
            let valid = status.valid;
            drop(session_config);
            if valid {
                runtime_config_status_from_values(effective_runtime_config_values(&runtime_config))
            } else {
                status
            }
        }
        Err(_) => runtime_config_status_from_input(&payload),
    }
}

#[tauri::command]
fn clear_runtime_config(runtime_config: State<'_, RuntimeConfigState>) -> RuntimeConfigStatus {
    if let Ok(mut session_config) = runtime_config.session_config.lock() {
        *session_config = None;
    }

    runtime_config_status_from_values(effective_runtime_config_values(&runtime_config))
}

pub fn run() {
    tauri::Builder::default()
        .manage(RuntimeConfigState::default())
        .setup(|app| {
            if let Err(error) = initialize_vector_database(app.handle()) {
                eprintln!("SQLite-vec initialization skipped: {error}");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            execute_local_gemma,
            dispatch_deep_research,
            get_active_model_identity,
            get_runtime_config_status,
            validate_runtime_config,
            apply_runtime_config,
            clear_runtime_config
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

async fn run_local_gemma(app: AppHandle, run_id: String, payload: ExecuteLocalGemmaRequest) {
    let config = match load_local_gemma_config(&app) {
        Ok(config) => config,
        Err(error) => {
            emit_local_error(&app, run_id, error);
            return;
        }
    };

    let context_packet = ContextPacket {
        prompt: payload.prompt,
        scalpel_context: payload.scalpel_context,
    };
    let request = build_openai_chat_request(&config, &context_packet);

    let client = match Client::builder().timeout(config.timeout).build() {
        Ok(client) => client,
        Err(_) => {
            emit_local_error(
                &app,
                run_id,
                "Could not initialize local model client.".to_string(),
            );
            return;
        }
    };

    match client
        .post(config.endpoint.clone())
        .json(&request)
        .send()
        .await
    {
        Ok(response) => {
            let status = response.status();
            let body = match response.text().await {
                Ok(body) => body,
                Err(_) => {
                    emit_local_error(
                        &app,
                        run_id,
                        "Could not read local model response.".to_string(),
                    );
                    return;
                }
            };

            if !status.is_success() {
                emit_local_error(
                    &app,
                    run_id,
                    format!(
                        "Local model endpoint returned HTTP {}: {}",
                        status.as_u16(),
                        short_response_snippet(&body)
                    ),
                );
                return;
            }

            match normalize_openai_response(&body) {
                Ok(ModelResponseEvent::ModelText(text)) => emit_local_update(
                    &app,
                    LocalGemmaUpdate {
                        run_id,
                        delta: text,
                        done: true,
                        error: None,
                    },
                ),
                Ok(ModelResponseEvent::ToolCallRequested {
                    approval_required: UserApprovalState::Required,
                }) => emit_local_error(
                    &app,
                    run_id,
                    "The local model requested a tool call, but tool execution is disabled in Phase 3."
                        .to_string(),
                ),
                Ok(ModelResponseEvent::ToolCallRequested { .. }) => emit_local_error(
                    &app,
                    run_id,
                    "The local model returned a tool-call response without text.".to_string(),
                ),
                Ok(ModelResponseEvent::Error(error)) | Err(error) => {
                    emit_local_error(&app, run_id, error)
                }
                Ok(ModelResponseEvent::Complete) => {
                    emit_local_error(&app, run_id, "Local model response was empty.".to_string())
                }
            }
        }
        Err(error) => {
            let message = if error.is_timeout() {
                "Local model request timed out.".to_string()
            } else if error.is_connect() {
                "Could not reach local model endpoint. Confirm the local model runtime is running."
                    .to_string()
            } else {
                "Local model request failed.".to_string()
            };
            emit_local_error(&app, run_id, message);
        }
    }
}

fn load_local_gemma_config(app: &AppHandle) -> Result<LocalGemmaConfig, String> {
    let runtime_config = app.state::<RuntimeConfigState>();
    local_gemma_config_from_values(effective_runtime_config_values(&runtime_config))
}

fn optional_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn effective_runtime_config_values(
    runtime_config: &RuntimeConfigState,
) -> EffectiveRuntimeConfigValues {
    effective_runtime_config_values_from_parts(
        runtime_config
            .session_config
            .lock()
            .ok()
            .and_then(|session_config| session_config.clone()),
        environment_runtime_config_values(),
    )
}

fn environment_runtime_config_values() -> EffectiveRuntimeConfigValues {
    let endpoint = optional_env("AI_UNDERGROUND_GEMMA_ENDPOINT");
    let model = optional_env("AI_UNDERGROUND_GEMMA_MODEL");
    let profile_id = optional_env("AI_UNDERGROUND_GEMMA_PROFILE");
    let source = if endpoint.is_some() || model.is_some() || profile_id.is_some() {
        RuntimeConfigSource::Environment
    } else {
        RuntimeConfigSource::Missing
    };

    EffectiveRuntimeConfigValues {
        source,
        endpoint,
        model,
        profile_id,
    }
}

fn effective_runtime_config_values_from_parts(
    session_config: Option<AppliedRuntimeConfig>,
    environment_config: EffectiveRuntimeConfigValues,
) -> EffectiveRuntimeConfigValues {
    match session_config {
        Some(config) => EffectiveRuntimeConfigValues {
            source: RuntimeConfigSource::Session,
            endpoint: Some(config.endpoint),
            model: Some(config.model),
            profile_id: config.profile_id,
        },
        None => environment_config,
    }
}

fn runtime_config_values_from_input(input: &RuntimeConfigInput) -> EffectiveRuntimeConfigValues {
    EffectiveRuntimeConfigValues {
        source: RuntimeConfigSource::Session,
        endpoint: trimmed_optional(&input.endpoint),
        model: trimmed_optional(&input.model),
        profile_id: input.profile_id.as_deref().and_then(trimmed_optional),
    }
}

fn trimmed_optional(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn applied_runtime_config_from_input(input: &RuntimeConfigInput) -> Option<AppliedRuntimeConfig> {
    let values = runtime_config_values_from_input(input);
    Some(AppliedRuntimeConfig {
        endpoint: values.endpoint?,
        model: values.model?,
        profile_id: values.profile_id,
    })
}

fn apply_runtime_config_to_session(
    session_config: &mut Option<AppliedRuntimeConfig>,
    input: &RuntimeConfigInput,
) -> RuntimeConfigStatus {
    let status = runtime_config_status_from_input(input);
    if !status.valid {
        return status;
    }

    if let Some(applied) = applied_runtime_config_from_input(input) {
        *session_config = Some(applied);
    }

    status
}

fn local_gemma_config_from_values(
    values: EffectiveRuntimeConfigValues,
) -> Result<LocalGemmaConfig, String> {
    let endpoint = values
        .endpoint
        .ok_or_else(|| "Local model endpoint is not configured.".to_string())?;
    let model = values
        .model
        .ok_or_else(|| "Local model id is not configured.".to_string())?;
    let endpoint = validate_local_model_endpoint(&endpoint)?;
    let profile = match values.profile_id {
        Some(profile_id) => Some(model_profile(&profile_id, &model)?),
        None => None,
    };
    let timeout = local_gemma_timeout();

    Ok(LocalGemmaConfig {
        endpoint,
        model,
        profile,
        timeout,
    })
}

fn active_model_identity_from_values(values: EffectiveRuntimeConfigValues) -> ActiveModelIdentity {
    let status = runtime_config_status_from_values(values.clone());

    if status.status != ActiveModelIdentityStatus::Configured {
        return inactive_model_identity(
            status.status,
            status.profile_id,
            &status.display_name,
            status.endpoint_configured,
            status.model_configured,
            status
                .detail
                .as_deref()
                .unwrap_or("Local model is not configured."),
        );
    }

    match (values.profile_id, values.model) {
        (Some(profile_id), Some(model)) => match model_profile(&profile_id, &model) {
            Ok(profile) => ActiveModelIdentity {
                status: ActiveModelIdentityStatus::Configured,
                profile_id: Some(profile.profile_id.to_string()),
                display_name: profile.display_name.to_string(),
                runtime_label: "OpenAI-compatible local runtime".to_string(),
                runtime_model_id: Some(profile.runtime_model_id),
                endpoint_configured: status.endpoint_configured,
                model_configured: status.model_configured,
                profile_known: true,
                detail: None,
            },
            Err(_) => inactive_model_identity(
                ActiveModelIdentityStatus::UnknownProfile,
                Some(profile_id),
                "Unknown Profile",
                status.endpoint_configured,
                status.model_configured,
                "Configured local model profile is not recognized.",
            ),
        },
        _ => inactive_model_identity(
            ActiveModelIdentityStatus::MissingProfile,
            None,
            "Profile Not Set",
            status.endpoint_configured,
            status.model_configured,
            "Select a local model profile.",
        ),
    }
}

fn runtime_config_status_from_input(input: &RuntimeConfigInput) -> RuntimeConfigStatus {
    runtime_config_status_from_values(runtime_config_values_from_input(input))
}

fn runtime_config_status_from_values(values: EffectiveRuntimeConfigValues) -> RuntimeConfigStatus {
    let endpoint_configured = values.endpoint.is_some();
    let model_configured = values.model.is_some();
    let sanitized_endpoint_label = sanitized_endpoint_label(values.endpoint.as_deref());

    let Some(endpoint) = values.endpoint else {
        return runtime_config_status(
            values.source,
            ActiveModelIdentityStatus::MissingEndpoint,
            "Not Configured",
            values.profile_id,
            endpoint_configured,
            model_configured,
            false,
            sanitized_endpoint_label,
            "Local runtime endpoint is not configured.",
        );
    };

    let Some(model) = values.model else {
        return runtime_config_status(
            values.source,
            ActiveModelIdentityStatus::MissingModel,
            "Not Configured",
            values.profile_id,
            endpoint_configured,
            model_configured,
            false,
            sanitized_endpoint_label,
            "Model ID is not configured.",
        );
    };

    if validate_local_model_endpoint(&endpoint).is_err() {
        return runtime_config_status(
            values.source,
            ActiveModelIdentityStatus::InvalidEndpoint,
            "Endpoint Blocked",
            values.profile_id,
            endpoint_configured,
            model_configured,
            false,
            sanitized_endpoint_label,
            "Endpoint must be localhost and target /v1/chat/completions.",
        );
    }

    let Some(profile_id) = values.profile_id else {
        return runtime_config_status(
            values.source,
            ActiveModelIdentityStatus::MissingProfile,
            "Profile Not Set",
            None,
            endpoint_configured,
            model_configured,
            false,
            sanitized_endpoint_label,
            "Select a local model profile.",
        );
    };

    match model_profile(&profile_id, &model) {
        Ok(profile) => RuntimeConfigStatus {
            source: values.source,
            valid: true,
            status: ActiveModelIdentityStatus::Configured,
            profile_id: Some(profile.profile_id.to_string()),
            display_name: profile.display_name.to_string(),
            endpoint_configured,
            model_configured,
            profile_known: true,
            sanitized_endpoint_label,
            detail: None,
        },
        Err(_) => runtime_config_status(
            values.source,
            ActiveModelIdentityStatus::UnknownProfile,
            "Unknown Profile",
            Some(profile_id),
            endpoint_configured,
            model_configured,
            false,
            sanitized_endpoint_label,
            "Configured local model profile is not recognized.",
        ),
    }
}

fn runtime_config_status(
    source: RuntimeConfigSource,
    status: ActiveModelIdentityStatus,
    display_name: &str,
    profile_id: Option<String>,
    endpoint_configured: bool,
    model_configured: bool,
    profile_known: bool,
    sanitized_endpoint_label: Option<String>,
    detail: &str,
) -> RuntimeConfigStatus {
    RuntimeConfigStatus {
        source,
        valid: false,
        status,
        display_name: display_name.to_string(),
        profile_id,
        endpoint_configured,
        model_configured,
        profile_known,
        sanitized_endpoint_label,
        detail: Some(detail.to_string()),
    }
}

fn sanitized_endpoint_label(endpoint: Option<&str>) -> Option<String> {
    endpoint.map(
        |raw_endpoint| match validate_local_model_endpoint(raw_endpoint) {
            Ok(endpoint) => match endpoint.host_str() {
                Some("localhost") => "localhost".to_string(),
                Some("127.0.0.1") => "127.0.0.1".to_string(),
                Some("::1") | Some("[::1]") => "[::1]".to_string(),
                _ => "local endpoint configured".to_string(),
            },
            Err(_) => "local endpoint rejected".to_string(),
        },
    )
}

fn inactive_model_identity(
    status: ActiveModelIdentityStatus,
    profile_id: Option<String>,
    display_name: &str,
    endpoint_configured: bool,
    model_configured: bool,
    detail: &str,
) -> ActiveModelIdentity {
    ActiveModelIdentity {
        status,
        profile_id,
        display_name: display_name.to_string(),
        runtime_label: "Local Runtime".to_string(),
        runtime_model_id: None,
        endpoint_configured,
        model_configured,
        profile_known: false,
        detail: Some(detail.to_string()),
    }
}

fn validate_local_model_endpoint(raw_endpoint: &str) -> Result<Url, String> {
    let endpoint = Url::parse(raw_endpoint)
        .map_err(|_| "Local model endpoint must be a valid localhost HTTP URL.".to_string())?;

    if endpoint.scheme() != "http" {
        return Err("Local model endpoint must use http on localhost.".to_string());
    }

    let Some(host) = endpoint.host_str() else {
        return Err("Local model endpoint must include a localhost host.".to_string());
    };

    if !matches!(host, "localhost" | "127.0.0.1" | "::1" | "[::1]") {
        return Err("Local model endpoint must target localhost or loopback only.".to_string());
    }

    if endpoint.path() != "/v1/chat/completions" {
        return Err("Local model endpoint must target /v1/chat/completions.".to_string());
    }

    Ok(endpoint)
}

fn local_gemma_timeout() -> Duration {
    std::env::var("AI_UNDERGROUND_GEMMA_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(LOCAL_GEMMA_DEFAULT_TIMEOUT_SECONDS))
}

fn build_openai_chat_request(
    config: &LocalGemmaConfig,
    context_packet: &ContextPacket,
) -> OpenAiChatCompletionRequest {
    let content = assemble_context_packet(context_packet);
    let _profile_id = config.profile.as_ref().map(|profile| profile.profile_id);

    OpenAiChatCompletionRequest {
        model: config.model.clone(),
        messages: vec![OpenAiChatMessage {
            role: "user",
            content,
        }],
        temperature: LOCAL_GEMMA_TEMPERATURE,
        stream: false,
    }
}

fn assemble_context_packet(context_packet: &ContextPacket) -> String {
    match context_packet
        .scalpel_context
        .as_ref()
        .map(|context| context.trim())
        .filter(|context| !context.is_empty())
    {
        Some(context) => format!(
            "Local context selected by the user:\n{context}\n\nUser prompt:\n{}",
            context_packet.prompt
        ),
        None => context_packet.prompt.clone(),
    }
}

fn model_profile(
    profile_id: &str,
    runtime_model_id: &str,
) -> Result<ModelCapabilityProfile, String> {
    match profile_id {
        "gemma4-26b-a4b" => Ok(ModelCapabilityProfile {
            profile_id: "gemma4-26b-a4b",
            display_name: "Gemma4-26B-MoE",
            runtime_model_id: runtime_model_id.to_string(),
            runtime_engine: RuntimeEngine::LmStudio,
            supports_text_input: CapabilitySupport::Yes,
            supports_image_input: CapabilitySupport::Yes,
            supports_video_frame_input: CapabilitySupport::Unknown,
            supports_function_calling: CapabilitySupport::Yes,
            supports_structured_output: CapabilitySupport::Unknown,
            supports_streaming: CapabilitySupport::Unknown,
            supports_embeddings: CapabilitySupport::Unknown,
            default_enabled_tools: vec![],
            risk_notes: "Runtime capability support must be verified in the loaded local engine.",
            explicit_uncensored_label: false,
        }),
        "gemma4-31b" => Ok(ModelCapabilityProfile {
            profile_id: "gemma4-31b",
            display_name: "Gemma4-31B Dense",
            runtime_model_id: runtime_model_id.to_string(),
            runtime_engine: RuntimeEngine::LmStudio,
            supports_text_input: CapabilitySupport::Unknown,
            supports_image_input: CapabilitySupport::Unknown,
            supports_video_frame_input: CapabilitySupport::Unknown,
            supports_function_calling: CapabilitySupport::Unknown,
            supports_structured_output: CapabilitySupport::Unknown,
            supports_streaming: CapabilitySupport::Unknown,
            supports_embeddings: CapabilitySupport::Unknown,
            default_enabled_tools: vec![],
            risk_notes: "Treat runtime/model-card capabilities as unknown until verified locally.",
            explicit_uncensored_label: false,
        }),
        "supergemma4-26b-uncensored" => Ok(ModelCapabilityProfile {
            profile_id: "supergemma4-26b-uncensored",
            display_name: "Gemma4-26B-Uncensored",
            runtime_model_id: runtime_model_id.to_string(),
            runtime_engine: RuntimeEngine::LmStudio,
            supports_text_input: CapabilitySupport::Yes,
            supports_image_input: CapabilitySupport::Unknown,
            supports_video_frame_input: CapabilitySupport::Unknown,
            supports_function_calling: CapabilitySupport::Unknown,
            supports_structured_output: CapabilitySupport::Unknown,
            supports_streaming: CapabilitySupport::Unknown,
            supports_embeddings: CapabilitySupport::Unknown,
            default_enabled_tools: vec![],
            risk_notes: "Explicit uncensored profile. Do not use as a silent default.",
            explicit_uncensored_label: true,
        }),
        _ => Err("Unknown local model profile.".to_string()),
    }
}

#[allow(dead_code)]
fn planned_tool_registry() -> Vec<ToolRegistryEntry> {
    vec![
        planned_tool(
            "web_search",
            "Web Search",
            "Remote-capable search tool. Disabled by default.",
            ToolLocation::RemoteCapable,
            ToolRiskLevel::High,
            "Requires clear enabled state and per-use approval.",
        ),
        planned_tool(
            "image_input",
            "Image Input",
            "User-selected image context for vision-capable local models.",
            ToolLocation::Local,
            ToolRiskLevel::Medium,
            "Requires attach-image control and model support indicator.",
        ),
        planned_tool(
            "file_context",
            "File Context",
            "User-selected local file context for the model.",
            ToolLocation::Local,
            ToolRiskLevel::Medium,
            "Requires explicit file selection and read-only state.",
        ),
        planned_tool(
            "artifact_create_update",
            "Artifact Create / Update",
            "Future model-assisted artifact drafting and editing seam.",
            ToolLocation::Local,
            ToolRiskLevel::Medium,
            "Requires approval before any write or artifact mutation.",
        ),
        planned_tool(
            "memory_lookup",
            "Memory Lookup",
            "Future local memory and context lookup seam.",
            ToolLocation::Local,
            ToolRiskLevel::Low,
            "Requires local-only memory availability state.",
        ),
        planned_tool(
            "local_workspace_query",
            "Local Workspace Query",
            "Future read-only local project awareness seam.",
            ToolLocation::Local,
            ToolRiskLevel::Medium,
            "Requires approval-gated read-only workspace scope.",
        ),
    ]
}

#[allow(dead_code)]
fn planned_tool(
    tool_id: &'static str,
    display_name: &'static str,
    description: &'static str,
    local_or_remote: ToolLocation,
    risk_level: ToolRiskLevel,
    ui_affordance_requirement: &'static str,
) -> ToolRegistryEntry {
    ToolRegistryEntry {
        tool_id,
        display_name,
        description,
        enabled: false,
        local_or_remote,
        requires_user_approval: true,
        input_schema: serde_json::json!({ "type": "object" }),
        output_schema: serde_json::json!({ "type": "object" }),
        supported_model_capabilities: vec![],
        risk_level,
        ui_affordance_requirement,
        implementation_status: ToolImplementationStatus::Planned,
    }
}

fn normalize_openai_response(body: &str) -> Result<ModelResponseEvent, String> {
    let parsed: OpenAiChatCompletionResponse = serde_json::from_str(body)
        .map_err(|_| "Local model response was not valid OpenAI-compatible JSON.".to_string())?;
    let Some(choice) = parsed.choices.first() else {
        return Err("Local model response did not include any choices.".to_string());
    };

    if let Some(message) = &choice.message {
        if message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty())
        {
            return Ok(ModelResponseEvent::ToolCallRequested {
                approval_required: UserApprovalState::Required,
            });
        }

        if let Some(content) = message.content.as_ref().map(|value| value.trim()) {
            if !content.is_empty() {
                return Ok(ModelResponseEvent::ModelText(content.to_string()));
            }
        }
    }

    if let Some(text) = choice.text.as_ref().map(|value| value.trim()) {
        if !text.is_empty() {
            return Ok(ModelResponseEvent::ModelText(text.to_string()));
        }
    }

    Err("Local model response did not include text content.".to_string())
}

fn short_response_snippet(body: &str) -> String {
    let collapsed = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return "empty response body".to_string();
    }

    const MAX_SNIPPET_LENGTH: usize = 240;
    if collapsed.chars().count() > MAX_SNIPPET_LENGTH {
        format!(
            "{}...",
            collapsed
                .chars()
                .take(MAX_SNIPPET_LENGTH)
                .collect::<String>()
        )
    } else {
        collapsed
    }
}

fn emit_local_error(app: &AppHandle, run_id: String, error: String) {
    emit_local_update(
        app,
        LocalGemmaUpdate {
            run_id,
            delta: String::new(),
            done: true,
            error: Some(error),
        },
    );
}

async fn run_deep_research_polling(
    app: AppHandle,
    task_id: String,
    payload: DispatchDeepResearchRequest,
) {
    let Some(endpoint) = std::env::var("GOOGLE_INTERACTIONS_API_ENDPOINT").ok() else {
        emit_research_update(
            &app,
            ResearchUpdate {
                task_id,
                query: payload.query,
                status: ResearchStatus::Failed,
                active_step: "Interactions API endpoint missing.".to_string(),
                thought_summary: None,
                steps: vec![],
                artifact: None,
                error: Some("GOOGLE_INTERACTIONS_API_ENDPOINT is not configured.".to_string()),
            },
        );
        return;
    };

    let client = Client::new();
    let api_key = std::env::var("GOOGLE_AI_STUDIO_API_KEY").ok();

    emit_research_update(
        &app,
        ResearchUpdate {
            task_id: task_id.clone(),
            query: payload.query.clone(),
            status: ResearchStatus::Running,
            active_step: "Dispatching Interactions API task.".to_string(),
            thought_summary: None,
            steps: vec![ResearchStep {
                title: Some("Dispatch".to_string()),
                active_step: Some("Dispatching Interactions API task.".to_string()),
                ..ResearchStep::default()
            }],
            artifact: None,
            error: None,
        },
    );

    let mut remote_task_id = task_id.clone();
    let dispatch_body = serde_json::json!({
        "task_id": task_id,
        "query": payload.query,
        "depth_mode": payload.depth_mode
    });

    let dispatch_result = authorized(
        client.post(&endpoint).json(&dispatch_body),
        api_key.as_deref(),
    )
    .send()
    .await;

    match dispatch_result {
        Ok(response) => match response.json::<GenericInteractionResponse>().await {
            Ok(body) => {
                if let Some(id) = body.task_id.or(body.id) {
                    remote_task_id = id;
                }
            }
            Err(error) => {
                emit_research_update(
                    &app,
                    ResearchUpdate {
                        task_id: remote_task_id,
                        query: payload.query,
                        status: ResearchStatus::Failed,
                        active_step: "Interactions API dispatch response was not valid JSON."
                            .to_string(),
                        thought_summary: None,
                        steps: vec![],
                        artifact: None,
                        error: Some(format!("Interactions API dispatch parse failed: {error}")),
                    },
                );
                return;
            }
        },
        Err(error) => {
            emit_research_update(
                &app,
                ResearchUpdate {
                    task_id: remote_task_id,
                    query: payload.query,
                    status: ResearchStatus::Failed,
                    active_step: "Interactions API dispatch request failed.".to_string(),
                    thought_summary: None,
                    steps: vec![],
                    artifact: None,
                    error: Some(format!("Interactions API dispatch failed: {error}")),
                },
            );
            return;
        }
    }

    loop {
        tokio::time::sleep(POLL_INTERVAL).await;

        let status_url = format!("{}/{}", endpoint.trim_end_matches('/'), remote_task_id);
        let status_result = authorized(client.get(status_url), api_key.as_deref())
            .send()
            .await;

        let update = match status_result {
            Ok(response) => match response.json::<GenericInteractionResponse>().await {
                Ok(body) => normalize_interaction_response(&remote_task_id, &payload.query, body),
                Err(error) => ResearchUpdate {
                    task_id: remote_task_id.clone(),
                    query: payload.query.clone(),
                    status: ResearchStatus::Failed,
                    active_step: "Interactions API status response was not valid JSON.".to_string(),
                    thought_summary: None,
                    steps: vec![],
                    artifact: None,
                    error: Some(format!("Interactions API status parse failed: {error}")),
                },
            },
            Err(error) => ResearchUpdate {
                task_id: remote_task_id.clone(),
                query: payload.query.clone(),
                status: ResearchStatus::Failed,
                active_step: "Interactions API status request failed.".to_string(),
                thought_summary: None,
                steps: vec![],
                artifact: None,
                error: Some(format!("Interactions API status failed: {error}")),
            },
        };

        let done = matches!(
            update.status,
            ResearchStatus::Completed | ResearchStatus::Failed
        );
        emit_research_update(&app, update);

        if done {
            break;
        }
    }
}

fn normalize_interaction_response(
    task_id: &str,
    query: &str,
    body: GenericInteractionResponse,
) -> ResearchUpdate {
    let status = body.status.unwrap_or(ResearchStatus::Running);
    let steps = body.steps.unwrap_or_default();
    let active_step = body
        .active_step
        .or_else(|| steps.iter().rev().find_map(|step| step.active_step.clone()))
        .unwrap_or_else(|| "Polling Interactions API task status.".to_string());

    ResearchUpdate {
        task_id: task_id.to_string(),
        query: query.to_string(),
        status,
        active_step,
        thought_summary: body.thought_summary,
        steps,
        artifact: body.artifact.or(body.output),
        error: body.error,
    }
}

fn authorized(builder: reqwest::RequestBuilder, api_key: Option<&str>) -> reqwest::RequestBuilder {
    match api_key {
        Some(key) if !key.trim().is_empty() => builder.bearer_auth(key),
        _ => builder,
    }
}

fn initialize_vector_database(app: &AppHandle) -> Result<(), String> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Could not resolve app data directory: {error}"))?;
    std::fs::create_dir_all(&data_dir)
        .map_err(|error| format!("Could not create app data directory: {error}"))?;

    let db_path = data_dir.join("the-ai-underground.sqlite3");
    open_sqlite_vec_connection(db_path)?;
    Ok(())
}

fn open_sqlite_vec_connection(db_path: PathBuf) -> Result<Connection, String> {
    let connection = Connection::open(db_path)
        .map_err(|error| format!("Could not open SQLite vector database: {error}"))?;

    if let Some(extension_path) = std::env::var("SQLITE_VEC_EXTENSION_PATH")
        .ok()
        .filter(|path| !path.trim().is_empty())
    {
        unsafe {
            connection
                .load_extension_enable()
                .map_err(|error| format!("Could not enable SQLite extension loading: {error}"))?;
            connection
                .load_extension(extension_path, None)
                .map_err(|error| format!("Could not load sqlite-vec extension: {error}"))?;
            connection
                .load_extension_disable()
                .map_err(|error| format!("Could not disable SQLite extension loading: {error}"))?;
        }
        connection
            .execute_batch(VECTOR_SCHEMA_SQL)
            .map_err(|error| format!("Could not initialize sqlite-vec schema: {error}"))?;
    } else {
        connection
            .execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS vector_memory_pending_vec_extension (
                  id TEXT PRIMARY KEY,
                  namespace TEXT NOT NULL,
                  embedding_dimension INTEGER NOT NULL DEFAULT 768,
                  raw_text TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                "#,
            )
            .map_err(|error| {
                format!("Could not initialize pending vector metadata table: {error}")
            })?;
    }

    connection
        .execute(
            "INSERT OR IGNORE INTO vector_memory_pending_vec_extension
             (id, namespace, embedding_dimension, raw_text, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            (
                "__sqlite_vec_schema__",
                "system",
                768_i64,
                VECTOR_SCHEMA_SQL,
                Utc::now().to_rfc3339(),
            ),
        )
        .ok();

    Ok(connection)
}

fn emit_local_update(app: &AppHandle, payload: LocalGemmaUpdate) {
    let _ = app.emit("local_gemma_update", payload);
}

fn emit_research_update(app: &AppHandle, payload: ResearchUpdate) {
    let _ = app.emit("research_update", payload);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_values(
        source: RuntimeConfigSource,
        endpoint: Option<&str>,
        model: Option<&str>,
        profile_id: Option<&str>,
    ) -> EffectiveRuntimeConfigValues {
        EffectiveRuntimeConfigValues {
            source,
            endpoint: endpoint.map(str::to_string),
            model: model.map(str::to_string),
            profile_id: profile_id.map(str::to_string),
        }
    }

    #[test]
    fn local_endpoint_validation_accepts_localhost() {
        let endpoint = validate_local_model_endpoint("http://localhost:1234/v1/chat/completions")
            .expect("localhost should be accepted");

        assert_eq!(endpoint.host_str(), Some("localhost"));
    }

    #[test]
    fn local_endpoint_validation_accepts_ipv4_loopback() {
        let endpoint = validate_local_model_endpoint("http://127.0.0.1:1234/v1/chat/completions")
            .expect("127.0.0.1 should be accepted");

        assert_eq!(endpoint.host_str(), Some("127.0.0.1"));
    }

    #[test]
    fn local_endpoint_validation_accepts_ipv6_loopback() {
        let endpoint = validate_local_model_endpoint("http://[::1]:1234/v1/chat/completions")
            .expect("[::1] should be accepted");

        assert!(endpoint.as_str().starts_with("http://[::1]:1234/"));
    }

    #[test]
    fn local_endpoint_validation_rejects_wrong_localhost_path() {
        let error = validate_local_model_endpoint("http://localhost:1234/api/generate")
            .expect_err("wrong local path should be rejected");

        assert!(error.contains("/v1/chat/completions"));
    }

    #[test]
    fn local_endpoint_validation_rejects_lan_private_ips() {
        let error = validate_local_model_endpoint("http://192.168.1.22:1234/v1/chat/completions")
            .expect_err("LAN/private IP should be rejected");

        assert!(error.contains("localhost or loopback"));
    }

    #[test]
    fn local_endpoint_validation_rejects_public_ips_and_remote_domains() {
        let public_error = validate_local_model_endpoint("http://8.8.8.8:1234/v1/chat/completions")
            .expect_err("public IP should be rejected");
        let remote_error = validate_local_model_endpoint("http://example.com/v1/chat/completions")
            .expect_err("remote domain should be rejected");

        assert!(public_error.contains("localhost or loopback"));
        assert!(remote_error.contains("localhost or loopback"));
    }

    #[test]
    fn local_endpoint_validation_rejects_malformed_urls() {
        let error =
            validate_local_model_endpoint("not a url").expect_err("malformed URL should fail");

        assert!(error.contains("valid localhost HTTP URL"));
    }

    #[test]
    fn openai_response_normalization_extracts_message_content() {
        let body = r#"{
          "choices": [
            {
              "message": {
                "role": "assistant",
                "content": "Local Gemma response"
              }
            }
          ]
        }"#;

        let event = normalize_openai_response(body).expect("message content should normalize");

        assert_eq!(
            event,
            ModelResponseEvent::ModelText("Local Gemma response".to_string())
        );
    }

    #[test]
    fn openai_response_normalization_detects_disabled_tool_calls() {
        let body = r#"{
          "choices": [
            {
              "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                  {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                      "name": "web_search",
                      "arguments": "{}"
                    }
                  }
                ]
              }
            }
          ]
        }"#;

        let event = normalize_openai_response(body).expect("tool call should be detected");

        assert_eq!(
            event,
            ModelResponseEvent::ToolCallRequested {
                approval_required: UserApprovalState::Required,
            }
        );
    }

    #[test]
    fn openai_response_normalization_prioritizes_tool_calls_over_content() {
        let body = r#"{
          "choices": [
            {
              "message": {
                "role": "assistant",
                "content": "Do not treat this as final text.",
                "tool_calls": [
                  {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                      "name": "local_workspace_query",
                      "arguments": "{}"
                    }
                  }
                ]
              }
            }
          ]
        }"#;

        let event = normalize_openai_response(body).expect("tool call should be detected first");

        assert_eq!(
            event,
            ModelResponseEvent::ToolCallRequested {
                approval_required: UserApprovalState::Required,
            }
        );
    }

    #[test]
    fn openai_response_normalization_returns_controlled_error_for_malformed_or_empty_response() {
        let malformed = normalize_openai_response("{");
        let empty = normalize_openai_response(r#"{ "choices": [] }"#);

        assert!(malformed.is_err());
        assert!(empty.is_err());
    }

    #[test]
    fn short_response_snippet_truncates_unicode_on_char_boundaries() {
        let body = "語 ".repeat(300);

        let snippet = short_response_snippet(&body);

        assert!(snippet.ends_with("..."));
        assert!(snippet.chars().count() <= 243);
    }

    #[test]
    fn model_profile_maps_approved_display_labels() {
        let gemma_moe = model_profile("gemma4-26b-a4b", "runtime-id").expect("profile should map");
        let gemma_dense = model_profile("gemma4-31b", "runtime-id").expect("profile should map");
        let supergemma =
            model_profile("supergemma4-26b-uncensored", "runtime-id").expect("profile should map");

        assert_eq!(gemma_moe.display_name, "Gemma4-26B-MoE");
        assert_eq!(gemma_dense.display_name, "Gemma4-31B Dense");
        assert_eq!(supergemma.display_name, "Gemma4-26B-Uncensored");
        assert!(supergemma.explicit_uncensored_label);
    }

    #[test]
    fn active_identity_returns_unknown_profile_state() {
        let identity = active_model_identity_from_values(test_values(
            RuntimeConfigSource::Session,
            Some("http://127.0.0.1:1234/v1/chat/completions"),
            Some("google/gemma-4-31b"),
            Some("unknown-profile"),
        ));

        assert_eq!(identity.status, ActiveModelIdentityStatus::UnknownProfile);
        assert_eq!(identity.display_name, "Unknown Profile");
        assert_eq!(identity.profile_id, Some("unknown-profile".to_string()));
        assert!(!identity.profile_known);
    }

    #[test]
    fn active_identity_returns_missing_profile_state() {
        let identity = active_model_identity_from_values(test_values(
            RuntimeConfigSource::Session,
            Some("http://127.0.0.1:1234/v1/chat/completions"),
            Some("google/gemma-4-31b"),
            None,
        ));

        assert_eq!(identity.status, ActiveModelIdentityStatus::MissingProfile);
        assert_eq!(identity.display_name, "Profile Not Set");
        assert_eq!(identity.profile_id, None);
        assert!(!identity.profile_known);
    }

    #[test]
    fn active_identity_returns_missing_endpoint_state() {
        let identity = active_model_identity_from_values(test_values(
            RuntimeConfigSource::Missing,
            None,
            Some("google/gemma-4-31b"),
            Some("gemma4-31b"),
        ));

        assert_eq!(identity.status, ActiveModelIdentityStatus::MissingEndpoint);
        assert_eq!(identity.display_name, "Not Configured");
        assert!(!identity.endpoint_configured);
        assert!(identity.model_configured);
        assert!(!identity.profile_known);
    }

    #[test]
    fn active_identity_returns_missing_model_state() {
        let identity = active_model_identity_from_values(test_values(
            RuntimeConfigSource::Environment,
            Some("http://127.0.0.1:1234/v1/chat/completions"),
            None,
            Some("gemma4-31b"),
        ));

        assert_eq!(identity.status, ActiveModelIdentityStatus::MissingModel);
        assert_eq!(identity.display_name, "Not Configured");
        assert!(identity.endpoint_configured);
        assert!(!identity.model_configured);
        assert!(!identity.profile_known);
    }

    #[test]
    fn active_identity_does_not_expose_invalid_endpoint_value() {
        let remote_endpoint = "http://example.com/v1/chat/completions";
        let identity = active_model_identity_from_values(test_values(
            RuntimeConfigSource::Session,
            Some(remote_endpoint),
            Some("google/gemma-4-31b"),
            Some("gemma4-31b"),
        ));

        assert_eq!(identity.status, ActiveModelIdentityStatus::InvalidEndpoint);
        assert_eq!(identity.display_name, "Endpoint Blocked");
        assert_ne!(identity.detail.as_deref(), Some(remote_endpoint));
        assert!(!identity
            .detail
            .unwrap_or_default()
            .contains(remote_endpoint));
    }

    #[test]
    fn active_identity_returns_configured_profile_state() {
        let identity = active_model_identity_from_values(test_values(
            RuntimeConfigSource::Session,
            Some("http://localhost:1234/v1/chat/completions"),
            Some("google/gemma-4-31b"),
            Some("gemma4-31b"),
        ));

        assert_eq!(identity.status, ActiveModelIdentityStatus::Configured);
        assert_eq!(identity.display_name, "Gemma4-31B Dense");
        assert_eq!(identity.runtime_label, "OpenAI-compatible local runtime");
        assert_eq!(
            identity.runtime_model_id,
            Some("google/gemma-4-31b".to_string())
        );
        assert!(identity.profile_known);
    }

    #[test]
    fn session_config_overrides_environment_config() {
        let session = Some(AppliedRuntimeConfig {
            endpoint: "http://localhost:1234/v1/chat/completions".to_string(),
            model: "session-model".to_string(),
            profile_id: Some("gemma4-31b".to_string()),
        });
        let environment = test_values(
            RuntimeConfigSource::Environment,
            Some("http://127.0.0.1:9999/v1/chat/completions"),
            Some("environment-model"),
            Some("gemma4-26b-a4b"),
        );

        let effective = effective_runtime_config_values_from_parts(session, environment);

        assert_eq!(effective.source, RuntimeConfigSource::Session);
        assert_eq!(effective.model, Some("session-model".to_string()));
        assert_eq!(effective.profile_id, Some("gemma4-31b".to_string()));
    }

    #[test]
    fn environment_config_is_fallback_without_session_config() {
        let environment = test_values(
            RuntimeConfigSource::Environment,
            Some("http://127.0.0.1:1234/v1/chat/completions"),
            Some("environment-model"),
            Some("gemma4-26b-a4b"),
        );

        let effective = effective_runtime_config_values_from_parts(None, environment);

        assert_eq!(effective.source, RuntimeConfigSource::Environment);
        assert_eq!(effective.model, Some("environment-model".to_string()));
    }

    #[test]
    fn clear_session_config_returns_to_environment_fallback_or_missing_state() {
        let environment = test_values(RuntimeConfigSource::Missing, None, None, None);

        let effective = effective_runtime_config_values_from_parts(None, environment);
        let status = runtime_config_status_from_values(effective);

        assert_eq!(status.source, RuntimeConfigSource::Missing);
        assert_eq!(status.status, ActiveModelIdentityStatus::MissingEndpoint);
        assert!(!status.valid);
    }

    #[test]
    fn invalid_endpoint_is_rejected_and_not_applied() {
        let input = RuntimeConfigInput {
            endpoint: "http://example.com/v1/chat/completions".to_string(),
            model: "google/gemma-4-31b".to_string(),
            profile_id: Some("gemma4-31b".to_string()),
        };
        let mut session_config = Some(AppliedRuntimeConfig {
            endpoint: "http://localhost:1234/v1/chat/completions".to_string(),
            model: "existing-model".to_string(),
            profile_id: Some("gemma4-31b".to_string()),
        });

        let status = apply_runtime_config_to_session(&mut session_config, &input);

        assert_eq!(status.status, ActiveModelIdentityStatus::InvalidEndpoint);
        assert!(!status.valid);
        assert_eq!(
            session_config.as_ref().map(|config| config.model.as_str()),
            Some("existing-model")
        );
    }

    #[test]
    fn invalid_endpoint_status_does_not_expose_raw_endpoint() {
        let raw_endpoint = "http://example.com/v1/chat/completions";
        let status = runtime_config_status_from_values(test_values(
            RuntimeConfigSource::Session,
            Some(raw_endpoint),
            Some("google/gemma-4-31b"),
            Some("gemma4-31b"),
        ));

        assert_eq!(status.status, ActiveModelIdentityStatus::InvalidEndpoint);
        assert_ne!(status.detail.as_deref(), Some(raw_endpoint));
        assert!(!status.detail.unwrap_or_default().contains(raw_endpoint));
        assert_eq!(
            status.sanitized_endpoint_label,
            Some("local endpoint rejected".to_string())
        );
    }

    #[test]
    fn local_gemma_config_uses_effective_config_values() {
        let config = local_gemma_config_from_values(test_values(
            RuntimeConfigSource::Session,
            Some("http://127.0.0.1:1234/v1/chat/completions"),
            Some("session-model"),
            Some("gemma4-31b"),
        ))
        .expect("effective config should load");

        assert_eq!(config.model, "session-model");
        assert_eq!(config.endpoint.host_str(), Some("127.0.0.1"));
        assert_eq!(
            config.profile.as_ref().map(|profile| profile.profile_id),
            Some("gemma4-31b")
        );
    }
}
