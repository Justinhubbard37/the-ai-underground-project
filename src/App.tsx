import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Cpu,
  Database,
  FileText,
  Network,
  Search,
  ShieldAlert,
  Sparkles,
  Terminal,
  X,
} from 'lucide-react';
import {
  applyRuntimeConfig,
  clearRuntimeConfig,
  dispatchDeepResearch,
  executeLocalGemma,
  getActiveModelIdentity,
  getRuntimeConfigStatus,
  validateRuntimeConfig,
} from './ipc/tauriCommands';
import { registerTauriEventHandlers } from './ipc/tauriEvents';
import { parseResearchCommand } from './domain/parsers';
import { useUndergroundStore } from './state/undergroundStore';
import type {
  ActiveModelIdentity,
  DispatchDeepResearchPayload,
  ExecuteLocalGemmaPayload,
  RuntimeConfigInput,
  RuntimeConfigStatus,
  TacticalMessage,
} from './domain/types';

const MODEL_PROFILE_OPTIONS = [
  { id: 'gemma4-26b-a4b', label: 'Gemma4-26B-MoE' },
  { id: 'gemma4-31b', label: 'Gemma4-31B Dense' },
  { id: 'supergemma4-26b-uncensored', label: 'Gemma4-26B-Uncensored' },
];

const LM_STUDIO_LOCAL_DEFAULT: RuntimeConfigInput = {
  endpoint: 'http://127.0.0.1:1234/v1/chat/completions',
  model: 'google/gemma-4-31b',
  profile_id: 'gemma4-31b',
};

export default function App() {
  const [tacticalInput, setTacticalInput] = useState('');
  const [scalpelContext, setScalpelContext] = useState('');
  const [expandedThoughts, setExpandedThoughts] = useState<Record<string, boolean>>({});
  const {
    local_tactical_stream,
    cloud_research_queue,
    artifact_canvas_state,
    active_model_identity,
    active_model_identity_status,
    active_model_identity_error,
    runtime_config_panel_open,
    runtime_config_draft,
    runtime_config_status,
    runtime_config_validation,
    runtime_config_load_status,
    runtime_config_error,
    addUserMessage,
    startLocalModelMessage,
    startResearchModelMessage,
    enqueueResearchTask,
    failModelMessage,
    setActiveModelIdentityLoading,
    setActiveModelIdentity,
    failActiveModelIdentity,
    setRuntimeConfigPanelOpen,
    updateRuntimeConfigDraft,
    setRuntimeConfigLoading,
    setRuntimeConfigStatus,
    setRuntimeConfigValidation,
    failRuntimeConfig,
  } = useUndergroundStore();

  const refreshRuntimeState = useCallback(
    async (isDisposed: () => boolean = () => false) => {
      setActiveModelIdentityLoading();
      setRuntimeConfigLoading();

      try {
        const [identity, status] = await Promise.all([getActiveModelIdentity(), getRuntimeConfigStatus()]);
        if (isDisposed()) {
          return;
        }

        setActiveModelIdentity(identity);
        setRuntimeConfigStatus(status);
      } catch (error) {
        if (isDisposed()) {
          return;
        }

        const message = normalizeError(error);
        failActiveModelIdentity(message);
        failRuntimeConfig(message);
      }
    },
    [
      failActiveModelIdentity,
      failRuntimeConfig,
      setActiveModelIdentity,
      setActiveModelIdentityLoading,
      setRuntimeConfigLoading,
      setRuntimeConfigStatus,
    ],
  );

  useEffect(() => {
    let disposed = false;
    let cleanup: (() => void) | undefined;

    const runCleanup = () => {
      cleanup?.();
      cleanup = undefined;
    };

    registerTauriEventHandlers()
      .then((nextCleanup) => {
        if (disposed) {
          nextCleanup();
          return;
        }

        cleanup = nextCleanup;
      })
      .catch((error) => {
        if (disposed) {
          return;
        }

        const id = crypto.randomUUID();
        startLocalModelMessage(id);
        failModelMessage(id, `Tauri event channel failed to initialize: ${String(error)}`);
      });

    return () => {
      disposed = true;
      runCleanup();
    };
  }, [failModelMessage, startLocalModelMessage]);

  useEffect(() => {
    let disposed = false;

    void refreshRuntimeState(() => disposed);

    return () => {
      disposed = true;
    };
  }, [refreshRuntimeState]);

  const activeResearchTask = useMemo(
    () => cloud_research_queue.find((task) => task.status === 'running' || task.status === 'queued') ?? null,
    [cloud_research_queue],
  );

  const toggleThought = (id: string) => {
    setExpandedThoughts((previous) => ({ ...previous, [id]: !previous[id] }));
  };

  const handleCanvasSelection = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim().length > 0) {
      setScalpelContext(selection.toString().trim());
    }
  };

  const handleSubmit = async () => {
    const command = tacticalInput.trim();
    if (!command) {
      return;
    }

    setTacticalInput('');
    addUserMessage(command);

    if (command.toLowerCase().startsWith('/research')) {
      await dispatchResearch(command);
      return;
    }

    await dispatchLocal(command);
  };

  const dispatchLocal = async (prompt: string) => {
    const payload: ExecuteLocalGemmaPayload = {
      prompt,
      scalpel_context: scalpelContext.trim() || null,
    };

    try {
      const started = await executeLocalGemma(payload);
      startLocalModelMessage(started.run_id);
    } catch (error) {
      const id = crypto.randomUUID();
      startLocalModelMessage(id);
      failModelMessage(id, normalizeError(error));
    }
  };

  const dispatchResearch = async (rawCommand: string) => {
    const payload: DispatchDeepResearchPayload = parseResearchCommand(rawCommand);

    try {
      const started = await dispatchDeepResearch(payload);
      enqueueResearchTask({
        task_id: started.task_id,
        query: payload.query,
        status: 'queued',
        active_step: 'Queued for Interactions API dispatch.',
      });
      startResearchModelMessage(started.task_id, payload.query);
    } catch (error) {
      const id = crypto.randomUUID();
      startLocalModelMessage(id);
      failModelMessage(id, normalizeError(error));
    }
  };

  const handleValidateRuntimeConfig = async () => {
    setRuntimeConfigLoading();
    try {
      const status = await validateRuntimeConfig(runtime_config_draft);
      setRuntimeConfigValidation(status);
    } catch (error) {
      failRuntimeConfig(normalizeError(error));
    }
  };

  const handleApplyRuntimeConfig = async () => {
    setRuntimeConfigLoading();
    try {
      const status = await applyRuntimeConfig(runtime_config_draft);
      setRuntimeConfigStatus(status);
      setRuntimeConfigValidation(status);
      if (status.valid) {
        await refreshRuntimeState();
        setRuntimeConfigPanelOpen(false);
      }
    } catch (error) {
      failRuntimeConfig(normalizeError(error));
    }
  };

  const handleClearRuntimeConfig = async () => {
    setRuntimeConfigLoading();
    try {
      const status = await clearRuntimeConfig();
      setRuntimeConfigStatus(status);
      setRuntimeConfigValidation(status);
      await refreshRuntimeState();
    } catch (error) {
      failRuntimeConfig(normalizeError(error));
    }
  };

  return (
    <div className="relative flex h-screen w-full gap-4 overflow-hidden bg-[#030305] p-4 font-mono text-sm text-[#a1a1aa] selection:bg-[#20bd90] selection:text-black">
      <div className="pointer-events-none absolute left-[-10%] top-[-20%] h-[60%] w-[60%] rounded-full bg-[#20bd90] opacity-10 blur-[180px] mix-blend-screen" />
      <div className="pointer-events-none absolute bottom-[-20%] right-[-10%] h-[60%] w-[60%] rounded-full bg-[#5994d7] opacity-[0.07] blur-[180px] mix-blend-screen" />

      <TelemetryPane tasks={cloud_research_queue} />

      <main className="relative z-10 flex h-full w-2/4 flex-col overflow-hidden rounded-3xl border border-white/[0.05] bg-white/[0.015] shadow-2xl backdrop-blur-2xl">
        <div className="z-20 flex items-center justify-between border-b border-white/[0.05] bg-black/20 p-5">
          <div className="relative flex items-center gap-3">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-white">Central Command</span>
            <span className="rounded border border-[#20bd90]/20 bg-[#20bd90]/10 px-2 py-0.5 text-[9px] font-bold tracking-wide text-[#20bd90]">
              Explicit Routing
            </span>
            <ActiveModelBadge
              identity={active_model_identity}
              status={active_model_identity_status}
              error={active_model_identity_error}
              onClick={() => setRuntimeConfigPanelOpen(!runtime_config_panel_open)}
            />
            {runtime_config_panel_open && (
              <RuntimeConfigPanel
                draft={runtime_config_draft}
                status={runtime_config_status}
                validation={runtime_config_validation}
                loadStatus={runtime_config_load_status}
                error={runtime_config_error}
                onDraftChange={updateRuntimeConfigDraft}
                onUseDefault={() => updateRuntimeConfigDraft(LM_STUDIO_LOCAL_DEFAULT)}
                onValidate={() => void handleValidateRuntimeConfig()}
                onApply={() => void handleApplyRuntimeConfig()}
                onClear={() => void handleClearRuntimeConfig()}
                onClose={() => setRuntimeConfigPanelOpen(false)}
              />
            )}
          </div>
          <Sparkles size={14} className="text-white/40" />
        </div>

        <div className="flex-1 space-y-6 overflow-y-auto p-6 pb-40">
          {local_tactical_stream.length === 0 ? (
            <EmptyStream />
          ) : (
            local_tactical_stream.map((message) => (
              <TacticalMessageBlock
                key={message.id}
                message={message}
                expanded={Boolean(expandedThoughts[message.id])}
                onToggle={() => toggleThought(message.id)}
              />
            ))
          )}
        </div>

        <div className="absolute bottom-6 left-6 right-6 z-20 flex flex-col gap-2">
          {scalpelContext && (
            <div className="mx-4 mb-2 flex items-start justify-between rounded-xl border border-[#20bd90]/30 bg-black/60 p-3 shadow-2xl backdrop-blur-xl animate-[fadeSlideUp_0.3s_ease-out]">
              <div>
                <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-[#20bd90] drop-shadow-[0_0_5px_rgba(32,189,144,0.3)]">
                  Target Context Extracted
                </span>
                <span className="line-clamp-2 pr-4 text-xs italic text-white/80">&ldquo;{scalpelContext}&rdquo;</span>
              </div>
              <button
                aria-label="Clear extracted context"
                onClick={() => setScalpelContext('')}
                className="rounded-full bg-white/5 p-1 text-white/40 transition-colors hover:text-white"
              >
                x
              </button>
            </div>
          )}

          <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-[#111116]/80 p-4 shadow-[0_8px_30px_rgb(0,0,0,0.5)] backdrop-blur-3xl transition-all focus-within:border-[#20bd90]/50 focus-within:bg-[#15151a]/90">
            <span className="text-xl leading-none text-[#20bd90] drop-shadow-[0_0_8px_rgba(32,189,144,0.5)]">*</span>
            <textarea
              className="w-full resize-none border-none bg-transparent py-1 text-sm text-[#e1f5f9] outline-none placeholder:text-white/30"
              rows={1}
              placeholder="Type for local Gemma, or prefix with /research for cloud Deep Research..."
              value={tacticalInput}
              onChange={(event) => setTacticalInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void handleSubmit();
                }
              }}
            />
            <button
              aria-label="Submit tactical command"
              onClick={() => void handleSubmit()}
              className="shrink-0 rounded-xl bg-white/10 p-2 text-white transition-colors hover:bg-white/20"
            >
              <ChevronRight size={18} />
            </button>
          </div>
        </div>
      </main>

      <ArtifactPane
        artifact={artifact_canvas_state}
        activeStep={activeResearchTask?.active_step ?? null}
        isLoading={Boolean(activeResearchTask)}
        onMouseUp={handleCanvasSelection}
      />
    </div>
  );
}

function TelemetryPane({ tasks }: { tasks: ReturnType<typeof useUndergroundStore.getState>['cloud_research_queue'] }) {
  return (
    <aside className="relative z-10 flex h-full w-1/4 flex-col overflow-hidden rounded-3xl border border-white/[0.05] bg-white/[0.02] shadow-2xl backdrop-blur-2xl">
      <div className="flex items-center justify-between border-b border-white/[0.05] bg-black/20 p-5">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-white">Telemetry</span>
        <Activity size={14} className="text-[#20bd90]" />
      </div>

      <div className="flex flex-1 flex-col gap-6 overflow-y-auto p-5">
        <div>
          <div className="mb-4 flex items-center gap-2 text-[#e1f5f9]">
            <Network size={14} className="text-[#20bd90]" />
            <span className="text-[10px] font-bold uppercase tracking-wider opacity-80">Deep Research Queue</span>
          </div>
          <div className="space-y-3">
            {tasks.length === 0 ? (
              <p className="text-xs leading-relaxed text-white/40">No cloud tasks dispatched.</p>
            ) : (
              tasks.map((task) => (
                <div key={task.task_id} className="flex flex-col gap-1 text-xs leading-relaxed">
                  <div className="flex items-center justify-between gap-3">
                    <span className="truncate text-white/60">{task.query}</span>
                    <span className={queueStatusClass(task.status)}>{task.status}</span>
                  </div>
                  <span className={task.status === 'running' ? 'text-[#20bd90]' : 'text-white/40'}>
                    {task.active_step}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="border-t border-white/[0.05] bg-black/20 p-5">
        <div className="mb-4 flex items-center gap-2 text-[#e1f5f9]">
          <Cpu size={14} className="opacity-70" />
          <span className="text-[10px] font-bold uppercase tracking-wider opacity-80">MCP Access Matrix</span>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-white/70">
              <FileText size={14} />
              <span className="text-xs">Filesystem</span>
            </div>
            <span className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-[10px] font-medium tracking-wide text-white/60">
              READ-ONLY
            </span>
          </div>
          <div className="flex items-center justify-between text-[#fa3205]">
            <div className="flex items-center gap-2 drop-shadow-[0_0_5px_rgba(250,50,5,0.3)]">
              <Database size={14} />
              <span className="text-xs font-medium">SQLite-Vec</span>
            </div>
            <div className="flex items-center gap-1">
              <ShieldAlert size={12} className="animate-pulse" />
              <span className="rounded-md border border-[#fa3205]/30 bg-[#fa3205]/10 px-2 py-1 text-[10px] font-bold tracking-wide text-[#fa3205]">
                R/W ACTIVE
              </span>
            </div>
          </div>
          <div className="flex items-center justify-between opacity-40">
            <div className="flex items-center gap-2 text-white/70">
              <Terminal size={14} />
              <span className="text-xs">Bash Executor</span>
            </div>
            <span className="rounded-md border border-white/5 bg-black/40 px-2 py-1 text-[10px] font-medium tracking-wide text-white/40">
              DISABLED
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}

function ActiveModelBadge({
  identity,
  status,
  error,
  onClick,
}: {
  identity: ActiveModelIdentity | null;
  status: 'idle' | 'loading' | 'ready' | 'error';
  error: string | null;
  onClick: () => void;
}) {
  const label = activeModelLabel(identity, status);
  const detail = activeModelDetail(identity, error);

  return (
    <button
      type="button"
      title={detail}
      onClick={onClick}
      className="rounded border border-[#5994d7]/20 bg-[#5994d7]/10 px-2 py-0.5 text-[9px] font-bold tracking-wide text-[#e1f5f9] transition-colors hover:border-[#5994d7]/40 hover:bg-[#5994d7]/20"
    >
      {label}
    </button>
  );
}

function RuntimeConfigPanel({
  draft,
  status,
  validation,
  loadStatus,
  error,
  onDraftChange,
  onUseDefault,
  onValidate,
  onApply,
  onClear,
  onClose,
}: {
  draft: RuntimeConfigInput;
  status: RuntimeConfigStatus | null;
  validation: RuntimeConfigStatus | null;
  loadStatus: 'idle' | 'loading' | 'ready' | 'error';
  error: string | null;
  onDraftChange: (patch: Partial<RuntimeConfigInput>) => void;
  onUseDefault: () => void;
  onValidate: () => void;
  onApply: () => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const feedback = validation ?? status;
  const busy = loadStatus === 'loading';

  return (
    <div className="absolute left-0 top-8 z-50 w-[420px] rounded-2xl border border-white/10 bg-[#07070b]/95 p-4 shadow-2xl backdrop-blur-2xl">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <span className="block text-[10px] font-bold uppercase tracking-widest text-[#20bd90]">
            Choose your local AI engine
          </span>
          <span className="mt-1 block text-[10px] uppercase tracking-wider text-white/40">Current session only</span>
        </div>
        <button
          type="button"
          aria-label="Close local runtime configuration"
          onClick={onClose}
          className="rounded-lg bg-white/5 p-1 text-white/40 transition-colors hover:text-white"
        >
          <X size={14} />
        </button>
      </div>

      <div className="space-y-3">
        <button
          type="button"
          disabled={busy}
          onClick={onUseDefault}
          className="w-full rounded-xl border border-[#20bd90]/25 bg-[#20bd90]/10 p-3 text-left transition-colors hover:border-[#20bd90]/45 hover:bg-[#20bd90]/15 disabled:opacity-50"
        >
          <span className="block text-xs font-bold text-[#e1f5f9]">Use LM Studio Local Default</span>
          <span className="mt-1 block text-[10px] leading-relaxed text-white/45">
            Fills the current-session settings for the standard local server.
          </span>
        </button>

        <label className="block">
          <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-white/50">Model profile</span>
          <select
            value={draft.profile_id ?? ''}
            onChange={(event) => onDraftChange({ profile_id: event.target.value || null })}
            className="w-full rounded-xl border border-white/10 bg-black/50 px-3 py-2 text-xs text-[#e1f5f9] outline-none transition-colors focus:border-[#20bd90]/50"
          >
            <option value="">Profile Not Set</option>
            {MODEL_PROFILE_OPTIONS.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.label}
              </option>
            ))}
          </select>
        </label>

        <div className="rounded-xl border border-white/10 bg-black/30 p-3">
          <span className="mb-3 block text-[10px] font-bold uppercase tracking-wider text-white/45">
            Advanced Runtime Settings
          </span>
          <div className="space-y-3">
            <RuntimeConfigField
              label="Local runtime endpoint"
              value={draft.endpoint}
              placeholder="http://127.0.0.1:1234/v1/chat/completions"
              onChange={(value) => onDraftChange({ endpoint: value })}
            />
            <RuntimeConfigField
              label="Model ID"
              value={draft.model}
              placeholder="google/gemma-4-31b"
              onChange={(value) => onDraftChange({ model: value })}
            />
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={busy}
          onClick={onValidate}
          className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-[10px] font-bold uppercase tracking-wide text-white/70 transition-colors hover:bg-white/10 disabled:opacity-50"
        >
          Validate
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={onApply}
          className="rounded-lg border border-[#20bd90]/30 bg-[#20bd90]/10 px-3 py-2 text-[10px] font-bold uppercase tracking-wide text-[#20bd90] transition-colors hover:bg-[#20bd90]/20 disabled:opacity-50"
        >
          Apply
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={onClear}
          className="rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-[10px] font-bold uppercase tracking-wide text-white/50 transition-colors hover:text-white disabled:opacity-50"
        >
          Clear Session Config
        </button>
      </div>

      <div className="mt-4 rounded-xl border border-white/5 bg-black/40 p-3 text-xs leading-relaxed text-white/50">
        {error ? (
          <span className="text-[#fa3205]">{error}</span>
        ) : feedback ? (
          <RuntimeConfigFeedback status={feedback} />
        ) : (
          <span>Local Model: Checking</span>
        )}
      </div>
    </div>
  );
}

function RuntimeConfigField({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  placeholder: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-white/50">{label}</span>
      <input
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-xl border border-white/10 bg-black/50 px-3 py-2 text-xs text-[#e1f5f9] outline-none transition-colors placeholder:text-white/25 focus:border-[#20bd90]/50"
      />
    </label>
  );
}

function RuntimeConfigFeedback({ status }: { status: RuntimeConfigStatus }) {
  const label = `Local Model: ${status.display_name}`;
  const source = status.source === 'session' ? 'Session' : status.source === 'environment' ? 'Environment' : 'Missing';

  return (
    <div className="space-y-1">
      <div className={status.valid ? 'text-[#20bd90]' : 'text-white/60'}>{label}</div>
      <div className="text-[10px] uppercase tracking-wider text-white/35">
        {source}
        {status.sanitized_endpoint_label ? ` / ${status.sanitized_endpoint_label}` : ''}
      </div>
      {status.detail && <div>{status.detail}</div>}
    </div>
  );
}

function activeModelLabel(
  identity: ActiveModelIdentity | null,
  status: 'idle' | 'loading' | 'ready' | 'error',
): string {
  if (status === 'loading' || status === 'idle') {
    return 'Local Model: Checking';
  }

  if (status === 'error') {
    return 'Local Model: Unavailable';
  }

  if (!identity) {
    return 'Local Model: Unknown';
  }

  return `Local Model: ${identity.display_name}`;
}

function activeModelDetail(identity: ActiveModelIdentity | null, error: string | null): string | undefined {
  if (error) {
    return error;
  }

  if (!identity) {
    return undefined;
  }

  return identity.detail ?? undefined;
}

function TacticalMessageBlock({
  message,
  expanded,
  onToggle,
}: {
  message: TacticalMessage;
  expanded: boolean;
  onToggle: () => void;
}) {
  if (message.role === 'user') {
    return (
      <div className="flex flex-col">
        <div className="self-end max-w-[80%]">
          <div className="mb-1.5 pr-2 text-right text-[10px] font-bold uppercase tracking-wider text-white/40">
            Director
          </div>
          <div className="rounded-2xl rounded-tr-sm border border-white/10 bg-white/10 p-4 leading-relaxed text-[#e1f5f9] shadow-lg backdrop-blur-md">
            {message.content}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      <div className="self-start max-w-[85%]">
        <div className="mb-1.5 flex items-center gap-2 pl-2 text-[10px] font-bold uppercase tracking-wider text-[#20bd90] drop-shadow-[0_0_5px_rgba(32,189,144,0.3)]">
          Model System
          {message.status === 'streaming' && <span className="h-1.5 w-1.5 rounded-full bg-[#20bd90] animate-pulse" />}
        </div>

        <div className="relative overflow-hidden rounded-2xl rounded-tl-sm border border-white/5 bg-black/40 p-5 text-[#e1f5f9] shadow-xl backdrop-blur-md">
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-[#20bd90]/5 to-transparent" />
          {message.thinking_content && (
            <div className="relative z-10 mb-4">
              <button
                onClick={onToggle}
                className="mb-3 flex items-center gap-1.5 rounded-lg border border-white/5 bg-white/5 px-3 py-1.5 text-xs text-white/40 transition-colors hover:text-white/80"
              >
                {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                <span className="tracking-wide">reasoning_content</span>
              </button>

              {expanded && (
                <div className="ml-2 whitespace-pre-wrap border-l border-white/10 py-2 pl-4 text-xs leading-relaxed text-white/50">
                  {message.thinking_content}
                </div>
              )}
            </div>
          )}

          <div className="relative z-10 whitespace-pre-wrap leading-relaxed">{message.content}</div>
        </div>
      </div>
    </div>
  );
}

function ArtifactPane({
  artifact,
  activeStep,
  isLoading,
  onMouseUp,
}: {
  artifact: string;
  activeStep: string | null;
  isLoading: boolean;
  onMouseUp: () => void;
}) {
  const renderedArtifact = useMemo(() => renderArtifactMarkup(artifact), [artifact]);

  return (
    <aside className="relative z-10 flex h-full w-1/4 flex-col overflow-hidden rounded-3xl border border-white/[0.05] bg-white/[0.02] shadow-2xl backdrop-blur-2xl">
      <div className="flex items-center justify-between border-b border-white/[0.05] bg-black/20 p-5">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-[#e1f5f9]">Artifact Workspace</span>
        <Search size={14} className="text-white/40" />
      </div>

      <div className="flex-1 overflow-y-auto scroll-smooth p-6" onMouseUp={onMouseUp}>
        {isLoading ? (
          <ResearchSkeleton activeStep={activeStep ?? 'Waiting for cloud task status.'} />
        ) : artifact ? (
          <div className="artifact-markup animate-[fadeSlideUp_0.6s_cubic-bezier(0.16,1,0.3,1)_forwards]">
            {renderedArtifact}
          </div>
        ) : (
          <DefaultArtifact />
        )}
      </div>
    </aside>
  );
}

function EmptyStream() {
  return (
    <div className="mx-auto mt-20 max-w-sm text-center text-xs leading-relaxed text-white/40">
      Standard text routes to local Gemma. Commands prefixed with <span className="text-[#20bd90]">/research</span>{' '}
      route directly to cloud Deep Research.
    </div>
  );
}

function ResearchSkeleton({ activeStep }: { activeStep: string }) {
  return (
    <div className="space-y-5">
      <div>
        <span className="mb-2 block text-[10px] font-bold uppercase tracking-wider text-[#20bd90]">
          Deep Research Running
        </span>
        <p className="text-xs leading-relaxed text-white/60">{activeStep}</p>
      </div>
      {[64, 112, 84, 140, 96].map((width, index) => (
        <div key={index} className="artifact-skeleton h-4 rounded bg-white/[0.06]" style={{ width: `${width}%` }} />
      ))}
      <div className="artifact-skeleton h-28 rounded-xl border border-white/10 bg-black/30" />
      {[88, 72, 100].map((width, index) => (
        <div key={index} className="artifact-skeleton h-4 rounded bg-white/[0.06]" style={{ width: `${width}%` }} />
      ))}
    </div>
  );
}

function DefaultArtifact() {
  return (
    <div className="artifact-markup animate-[fadeSlideUp_0.6s_cubic-bezier(0.16,1,0.3,1)_forwards]">
      <h2>
        Deep Research Payload:
        <br />
        <span className="font-bold text-[#20bd90]">Awaiting Dispatch</span>
      </h2>
      <p>
        Completed Interactions API tasks will render here. Select text in this workspace to inject it into the next
        local Gemma command as scalpel context.
      </p>
      <div className="mt-8 flex cursor-crosshair flex-col items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-black/40 p-6 transition-colors hover:border-[#20bd90]/30">
        <div className="flex h-16 w-16 animate-[spin_12s_linear_infinite] items-center justify-center rounded-full border border-[#20bd90]/50 shadow-[0_0_15px_rgba(32,189,144,0.2)]">
          <div className="h-6 w-6 rotate-45 border border-[#e1f5f9]/80 bg-white/5 backdrop-blur-sm" />
        </div>
        <span className="mt-4 text-[10px] uppercase tracking-widest text-white/40">Artifact Canvas</span>
      </div>
    </div>
  );
}

function renderArtifactMarkup(input: string): JSX.Element[] {
  return input
    .trim()
    .split(/\n\s*\n/)
    .filter((block) => block.trim().length > 0)
    .map((block, index) => renderArtifactBlock(block, index));
}

function renderArtifactBlock(block: string, index: number): JSX.Element {
  const trimmed = block.trim();
  const heading = /^(#{1,3})\s+(.+)$/.exec(trimmed);

  if (heading) {
    const [, marker, text] = heading;
    const children = renderInlineCode(text);

    if (marker === '#') {
      return <h1 key={index}>{children}</h1>;
    }

    if (marker === '##') {
      return <h2 key={index}>{children}</h2>;
    }

    return <h3 key={index}>{children}</h3>;
  }

  const lines = block.split(/\r?\n/);
  return (
    <p key={index}>
      {lines.map((line, lineIndex) => (
        <span key={lineIndex}>
          {lineIndex > 0 && <br />}
          {renderInlineCode(line)}
        </span>
      ))}
    </p>
  );
}

function renderInlineCode(text: string): Array<string | JSX.Element> {
  const parts = text.split(/(`[^`]+`)/g);

  return parts
    .filter((part) => part.length > 0)
    .map((part, index) => {
      if (part.startsWith('`') && part.endsWith('`') && part.length > 1) {
        return <code key={index}>{part.slice(1, -1)}</code>;
      }

      return part;
    });
}

function queueStatusClass(status: string): string {
  if (status === 'running') {
    return 'shrink-0 rounded bg-[#20bd90]/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-[#20bd90]';
  }

  if (status === 'failed') {
    return 'shrink-0 rounded bg-[#fa3205]/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-[#fa3205]';
  }

  return 'shrink-0 rounded bg-white/5 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-white/50';
}

function normalizeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}
