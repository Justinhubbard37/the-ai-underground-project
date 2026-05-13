import { create } from 'zustand';
import type { ActiveModelIdentity, ResearchTask, ResearchUpdate, TacticalMessage } from '../domain/types';
import { extractThoughtSummary } from '../domain/parsers';

type ActiveModelIdentityLoadStatus = 'idle' | 'loading' | 'ready' | 'error';

interface UndergroundState {
  local_tactical_stream: TacticalMessage[];
  cloud_research_queue: ResearchTask[];
  artifact_canvas_state: string;
  active_model_identity: ActiveModelIdentity | null;
  active_model_identity_status: ActiveModelIdentityLoadStatus;
  active_model_identity_error: string | null;
  addUserMessage: (content: string) => void;
  startLocalModelMessage: (id: string) => void;
  startResearchModelMessage: (taskId: string, query: string) => void;
  appendModelDelta: (id: string, contentDelta: string, thinkingDelta: string) => void;
  completeModelMessage: (id: string, fallbackContent?: string) => void;
  failModelMessage: (id: string, error: string) => void;
  enqueueResearchTask: (task: ResearchTask) => void;
  applyResearchUpdate: (update: ResearchUpdate) => void;
  setActiveModelIdentityLoading: () => void;
  setActiveModelIdentity: (identity: ActiveModelIdentity) => void;
  failActiveModelIdentity: (error: string) => void;
}

export const useUndergroundStore = create<UndergroundState>((set) => ({
  local_tactical_stream: [],
  cloud_research_queue: [],
  artifact_canvas_state: '',
  active_model_identity: null,
  active_model_identity_status: 'idle',
  active_model_identity_error: null,

  addUserMessage: (content) =>
    set((state) => ({
      local_tactical_stream: [
        ...state.local_tactical_stream,
        {
          id: crypto.randomUUID(),
          role: 'user',
          content,
          thinking_content: null,
          status: 'complete',
        },
      ],
    })),

  startLocalModelMessage: (id) =>
    set((state) => ({
      local_tactical_stream: [
        ...state.local_tactical_stream,
        {
          id,
          role: 'model',
          content: '',
          thinking_content: null,
          status: 'streaming',
        },
      ],
    })),

  startResearchModelMessage: (taskId, query) =>
    set((state) => ({
      local_tactical_stream: [
        ...state.local_tactical_stream,
        {
          id: researchMessageId(taskId),
          role: 'model',
          content: `Deep Research dispatched: ${query}`,
          thinking_content: 'Queued for Interactions API polling.',
          status: 'streaming',
        },
      ],
    })),

  appendModelDelta: (id, contentDelta, thinkingDelta) =>
    set((state) => ({
      local_tactical_stream: state.local_tactical_stream.map((message) => {
        if (message.id !== id) {
          return message;
        }

        return {
          ...message,
          content: message.content + contentDelta,
          thinking_content:
            thinkingDelta.length > 0 ? `${message.thinking_content ?? ''}${thinkingDelta}` : message.thinking_content,
        };
      }),
    })),

  completeModelMessage: (id, fallbackContent = '') =>
    set((state) => ({
      local_tactical_stream: state.local_tactical_stream.map((message) => {
        if (message.id !== id) {
          return message;
        }

        return {
          ...message,
          content: message.content || fallbackContent,
          status: 'complete',
        };
      }),
    })),

  failModelMessage: (id, error) =>
    set((state) => ({
      local_tactical_stream: state.local_tactical_stream.map((message) => {
        if (message.id !== id) {
          return message;
        }

        return {
          ...message,
          content: error,
          status: 'complete',
        };
      }),
    })),

  enqueueResearchTask: (task) =>
    set((state) => ({
      cloud_research_queue: upsertTask(state.cloud_research_queue, task),
    })),

  applyResearchUpdate: (update) =>
    set((state) => {
      const activeTask: ResearchTask = {
        task_id: update.task_id,
        query: update.query,
        status: update.status,
        active_step: update.active_step,
      };
      const thoughtSummary = extractThoughtSummary(update.steps, update.thought_summary);
      const modelId = researchMessageId(update.task_id);

      return {
        cloud_research_queue: upsertTask(state.cloud_research_queue, activeTask),
        artifact_canvas_state: update.artifact ?? state.artifact_canvas_state,
        local_tactical_stream: state.local_tactical_stream.map((message) => {
          if (message.id !== modelId) {
            return message;
          }

          return {
            ...message,
            thinking_content: thoughtSummary ?? update.active_step,
            content:
              update.status === 'completed'
                ? (update.artifact ?? 'Deep Research completed without an artifact payload.')
                : (update.error ?? `Deep Research ${update.status}: ${update.active_step}`),
            status: update.status === 'completed' || update.status === 'failed' ? 'complete' : 'streaming',
          };
        }),
      };
    }),

  setActiveModelIdentityLoading: () =>
    set({
      active_model_identity_status: 'loading',
      active_model_identity_error: null,
    }),

  setActiveModelIdentity: (identity) =>
    set({
      active_model_identity: identity,
      active_model_identity_status: 'ready',
      active_model_identity_error: null,
    }),

  failActiveModelIdentity: (error) =>
    set({
      active_model_identity: null,
      active_model_identity_status: 'error',
      active_model_identity_error: error,
    }),
}));

function upsertTask(tasks: ResearchTask[], nextTask: ResearchTask): ResearchTask[] {
  const existing = tasks.findIndex((task) => task.task_id === nextTask.task_id);
  if (existing === -1) {
    return [...tasks, nextTask];
  }

  return tasks.map((task) => (task.task_id === nextTask.task_id ? nextTask : task));
}

export function researchMessageId(taskId: string): string {
  return `research:${taskId}`;
}
