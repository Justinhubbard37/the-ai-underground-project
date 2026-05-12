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
