import { invoke } from '@tauri-apps/api/core';
import type {
  ActiveModelIdentity,
  DeepResearchStarted,
  DispatchDeepResearchPayload,
  ExecuteLocalGemmaPayload,
  LocalRunStarted,
  RuntimeConfigInput,
  RuntimeConfigStatus,
} from '../domain/types';
import {
  activeModelIdentitySchema,
  dispatchDeepResearchPayloadSchema,
  executeLocalGemmaPayloadSchema,
  runtimeConfigInputSchema,
  runtimeConfigStatusSchema,
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

export async function getRuntimeConfigStatus(): Promise<RuntimeConfigStatus> {
  const response = await invoke<RuntimeConfigStatus>('get_runtime_config_status');
  return runtimeConfigStatusSchema.parse(response);
}

export async function validateRuntimeConfig(input: RuntimeConfigInput): Promise<RuntimeConfigStatus> {
  const payload = runtimeConfigInputSchema.parse(input);
  const response = await invoke<RuntimeConfigStatus>('validate_runtime_config', { payload });
  return runtimeConfigStatusSchema.parse(response);
}

export async function applyRuntimeConfig(input: RuntimeConfigInput): Promise<RuntimeConfigStatus> {
  const payload = runtimeConfigInputSchema.parse(input);
  const response = await invoke<RuntimeConfigStatus>('apply_runtime_config', { payload });
  return runtimeConfigStatusSchema.parse(response);
}

export async function clearRuntimeConfig(): Promise<RuntimeConfigStatus> {
  const response = await invoke<RuntimeConfigStatus>('clear_runtime_config');
  return runtimeConfigStatusSchema.parse(response);
}

export async function clearSavedRuntimeConfig(): Promise<RuntimeConfigStatus> {
  const response = await invoke<RuntimeConfigStatus>('clear_saved_runtime_config');
  return runtimeConfigStatusSchema.parse(response);
}
