use std::path::PathBuf;
use std::time::Duration;

use chrono::Utc;
use reqwest::Client;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;

const LOCAL_GEMMA_TEMPERATURE: f32 = 0.2;
const LOCAL_GEMMA_SYSTEM_PROMPT: &str =
    "You are the local tactical model for The AI Underground. Keep routing local and never call cloud tools.";
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

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if let Err(error) = initialize_vector_database(app.handle()) {
                eprintln!("SQLite-vec initialization skipped: {error}");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            execute_local_gemma,
            dispatch_deep_research
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

async fn run_local_gemma(app: AppHandle, run_id: String, payload: ExecuteLocalGemmaRequest) {
    let Some(endpoint) = std::env::var("AI_UNDERGROUND_GEMMA_ENDPOINT").ok() else {
        emit_local_update(
            &app,
            LocalGemmaUpdate {
                run_id,
                delta: String::new(),
                done: true,
                error: Some(
                    "AI_UNDERGROUND_GEMMA_ENDPOINT is not configured for local Gemma streaming."
                        .to_string(),
                ),
            },
        );
        return;
    };

    let client = Client::new();
    let request = serde_json::json!({
        "prompt": payload.prompt,
        "scalpel_context": payload.scalpel_context,
        "temperature": LOCAL_GEMMA_TEMPERATURE,
        "system_prompt": LOCAL_GEMMA_SYSTEM_PROMPT
    });

    match client.post(endpoint).json(&request).send().await {
        Ok(response) => match response.text().await {
            Ok(text) => {
                emit_local_update(
                    &app,
                    LocalGemmaUpdate {
                        run_id,
                        delta: text,
                        done: true,
                        error: None,
                    },
                );
            }
            Err(error) => emit_local_update(
                &app,
                LocalGemmaUpdate {
                    run_id,
                    delta: String::new(),
                    done: true,
                    error: Some(format!("Local Gemma response read failed: {error}")),
                },
            ),
        },
        Err(error) => emit_local_update(
            &app,
            LocalGemmaUpdate {
                run_id,
                delta: String::new(),
                done: true,
                error: Some(format!("Local Gemma request failed: {error}")),
            },
        ),
    }
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
