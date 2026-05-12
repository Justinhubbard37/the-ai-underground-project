import { useEffect, useMemo, useState } from 'react';
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
} from 'lucide-react';
import { executeLocalGemma, dispatchDeepResearch } from './ipc/tauriCommands';
import { registerTauriEventHandlers } from './ipc/tauriEvents';
import { parseResearchCommand } from './domain/parsers';
import { useUndergroundStore } from './state/undergroundStore';
import type { DispatchDeepResearchPayload, ExecuteLocalGemmaPayload, TacticalMessage } from './domain/types';

export default function App() {
  const [tacticalInput, setTacticalInput] = useState('');
  const [scalpelContext, setScalpelContext] = useState('');
  const [expandedThoughts, setExpandedThoughts] = useState<Record<string, boolean>>({});
  const {
    local_tactical_stream,
    cloud_research_queue,
    artifact_canvas_state,
    addUserMessage,
    startLocalModelMessage,
    startResearchModelMessage,
    enqueueResearchTask,
    failModelMessage,
  } = useUndergroundStore();

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    registerTauriEventHandlers()
      .then((cleanup) => {
        unlisten = cleanup;
      })
      .catch((error) => {
        const id = crypto.randomUUID();
        startLocalModelMessage(id);
        failModelMessage(id, `Tauri event channel failed to initialize: ${String(error)}`);
      });

    return () => {
      unlisten?.();
    };
  }, [failModelMessage, startLocalModelMessage]);

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

  return (
    <div className="relative flex h-screen w-full gap-4 overflow-hidden bg-[#030305] p-4 font-mono text-sm text-[#a1a1aa] selection:bg-[#20bd90] selection:text-black">
      <div className="pointer-events-none absolute left-[-10%] top-[-20%] h-[60%] w-[60%] rounded-full bg-[#20bd90] opacity-10 blur-[180px] mix-blend-screen" />
      <div className="pointer-events-none absolute bottom-[-20%] right-[-10%] h-[60%] w-[60%] rounded-full bg-[#5994d7] opacity-[0.07] blur-[180px] mix-blend-screen" />

      <TelemetryPane tasks={cloud_research_queue} />

      <main className="relative z-10 flex h-full w-2/4 flex-col overflow-hidden rounded-3xl border border-white/[0.05] bg-white/[0.015] shadow-2xl backdrop-blur-2xl">
        <div className="z-20 flex items-center justify-between border-b border-white/[0.05] bg-black/20 p-5">
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-white">Central Command</span>
            <span className="rounded border border-[#20bd90]/20 bg-[#20bd90]/10 px-2 py-0.5 text-[9px] font-bold tracking-wide text-[#20bd90]">
              Explicit Routing
            </span>
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
