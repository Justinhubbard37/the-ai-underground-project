export type TacticalRole = 'user' | 'model';
export type TacticalStatus = 'streaming' | 'complete';

export interface TacticalMessage {
  id: string;
  role: TacticalRole;
  content: string;
  thinking_content: string | null;
  status: TacticalStatus;
}

export type ResearchStatus = 'queued' | 'running' | 'completed' | 'failed';

export interface ResearchTask {
  task_id: string;
  query: string;
  status: ResearchStatus;
  active_step: string;
}

export type DepthMode = 'standard' | 'max';

export interface ExecuteLocalGemmaPayload {
  prompt: string;
  scalpel_context: string | null;
}

export interface DispatchDeepResearchPayload {
  query: string;
  depth_mode: DepthMode;
}

export interface LocalRunStarted {
  run_id: string;
}

export interface DeepResearchStarted {
  task_id: string;
}

export type ActiveModelIdentityStatus =
  | 'configured'
  | 'missing_endpoint'
  | 'missing_model'
  | 'missing_profile'
  | 'unknown_profile'
  | 'invalid_endpoint';

export interface ActiveModelIdentity {
  status: ActiveModelIdentityStatus;
  profile_id: string | null;
  display_name: string;
  runtime_label: string;
  runtime_model_id: string | null;
  endpoint_configured: boolean;
  model_configured: boolean;
  profile_known: boolean;
  detail: string | null;
}

export type RuntimeConfigSource = 'session' | 'saved' | 'environment' | 'missing';

export interface RuntimeConfigInput {
  endpoint: string;
  model: string;
  profile_id: string | null;
}

export interface RuntimeConfigStatus {
  source: RuntimeConfigSource;
  valid: boolean;
  status: ActiveModelIdentityStatus;
  display_name: string;
  profile_id: string | null;
  endpoint_configured: boolean;
  model_configured: boolean;
  profile_known: boolean;
  sanitized_endpoint_label: string | null;
  detail: string | null;
  saved_config_problem: string | null;
}

export interface LocalGemmaUpdate {
  run_id: string;
  delta: string;
  done: boolean;
  error: string | null;
}

export interface ResearchStep {
  id?: string;
  title?: string;
  active_step?: string;
  thought_summary?: string;
  status?: ResearchStatus;
}

export interface ResearchUpdate {
  task_id: string;
  query: string;
  status: ResearchStatus;
  active_step: string;
  thought_summary: string | null;
  steps: ResearchStep[];
  artifact: string | null;
  error: string | null;
}
