import { invoke } from '@tauri-apps/api/core';
import type {
  ActiveModelIdentity,
  DeepResearchStarted,
  DispatchDeepResearchPayload,
  ExecuteLocalGemmaPayload,
  LocalRunStarted,
} from '../domain/types';
import {
  activeModelIdentitySchema,
  dispatchDeepResearchPayloadSchema,
  executeLocalGemmaPayloadSchema,
} from '../domain/schemas';

export async function executeLocalGemma(payload: ExecuteLocalGemmaPayload): Promise<LocalRunStarted> {
  const validated = executeLocalGemmaPayloadSchema.parse(payload);
  return invoke<LocalRunStarted>('execute_local_gemma', { payload: validated });
}

export async function dispatchDeepResearch(payload: DispatchDeepResearchPayload): Promise<DeepResearchStarted> {
  const validated = dispatchDeepResearchPayloadSchema.parse(payload);
  return invoke<DeepResearchStarted>('dispatch_deep_research', { payload: validated });
}

export async function getActiveModelIdentity(): Promise<ActiveModelIdentity> {
  const response = await invoke<ActiveModelIdentity>('get_active_model_identity');
  return activeModelIdentitySchema.parse(response);
}
