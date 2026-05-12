import { listen } from '@tauri-apps/api/event';
import { z } from 'zod';
import { localGemmaUpdateSchema, researchUpdateSchema } from '../domain/schemas';
import {
  createGemmaParserState,
  flushGemmaParserState,
  parseGemmaDelta,
  type GemmaParserState,
} from '../domain/parsers';
import { useUndergroundStore } from '../state/undergroundStore';

const parserStates = new Map<string, GemmaParserState>();

export async function registerTauriEventHandlers(): Promise<() => void> {
  const unlistenLocal = await listen<unknown>('local_gemma_update', (event) => {
    const parsed = localGemmaUpdateSchema.safeParse(event.payload);
    if (!parsed.success) {
      appendValidationError('Invalid local_gemma_update event payload.', parsed.error);
      return;
    }

    const update = parsed.data;
    const store = useUndergroundStore.getState();

    if (update.error) {
      parserStates.delete(update.run_id);
      store.failModelMessage(update.run_id, update.error);
      return;
    }

    const currentState = parserStates.get(update.run_id) ?? createGemmaParserState();
    const normalized = parseGemmaDelta(update.delta, currentState);
    parserStates.set(update.run_id, normalized.state);
    store.appendModelDelta(update.run_id, normalized.contentDelta, normalized.thinkingDelta);

    if (update.done) {
      const flushed = flushGemmaParserState(normalized.state);
      store.appendModelDelta(update.run_id, flushed.contentDelta, flushed.thinkingDelta);
      parserStates.delete(update.run_id);
      store.completeModelMessage(update.run_id, 'Local Gemma completed without visible content.');
    }
  });

  const unlistenResearch = await listen<unknown>('research_update', (event) => {
    const parsed = researchUpdateSchema.safeParse(event.payload);
    if (!parsed.success) {
      appendValidationError('Invalid research_update event payload.', parsed.error);
      return;
    }

    useUndergroundStore.getState().applyResearchUpdate(parsed.data);
  });

  return () => {
    unlistenLocal();
    unlistenResearch();
  };
}

function appendValidationError(message: string, error: z.ZodError): void {
  const detail = error.issues.map((issue) => issue.message).join(' ');
  const id = crypto.randomUUID();
  const store = useUndergroundStore.getState();
  store.startLocalModelMessage(id);
  store.failModelMessage(id, `${message} ${detail}`.trim());
}
