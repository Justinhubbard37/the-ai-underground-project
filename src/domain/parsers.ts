import type { DepthMode, ResearchStep } from './types';

const THOUGHT_OPEN = '<|channel>thought';
const THOUGHT_CLOSE = '<channel|>';

export interface GemmaParserState {
  inThinking: boolean;
  buffer: string;
}

export interface ParsedGemmaDelta {
  contentDelta: string;
  thinkingDelta: string;
  state: GemmaParserState;
}

export function createGemmaParserState(): GemmaParserState {
  return {
    inThinking: false,
    buffer: '',
  };
}

export function parseGemmaDelta(chunk: string, state: GemmaParserState): ParsedGemmaDelta {
  let input = state.buffer + chunk;
  let inThinking = state.inThinking;
  let contentDelta = '';
  let thinkingDelta = '';

  while (input.length > 0) {
    const marker = inThinking ? THOUGHT_CLOSE : THOUGHT_OPEN;
    const markerIndex = input.indexOf(marker);

    if (markerIndex === -1) {
      const partialMarkerLength = longestMarkerPrefixAtEnd(input, marker);
      const ready = input.slice(0, input.length - partialMarkerLength);
      if (inThinking) {
        thinkingDelta += ready;
      } else {
        contentDelta += ready;
      }
      input = input.slice(input.length - partialMarkerLength);
      break;
    }

    const beforeMarker = input.slice(0, markerIndex);
    if (inThinking) {
      thinkingDelta += beforeMarker;
    } else {
      contentDelta += beforeMarker;
    }

    input = input.slice(markerIndex + marker.length);
    inThinking = !inThinking;
  }

  return {
    contentDelta,
    thinkingDelta,
    state: {
      inThinking,
      buffer: input,
    },
  };
}

export function flushGemmaParserState(state: GemmaParserState): ParsedGemmaDelta {
  return {
    contentDelta: state.inThinking ? '' : state.buffer,
    thinkingDelta: state.inThinking ? state.buffer : '',
    state: {
      inThinking: state.inThinking,
      buffer: '',
    },
  };
}

export function extractThoughtSummary(steps: ResearchStep[], fallback: string | null): string | null {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const summary = steps[index]?.thought_summary?.trim();
    if (summary) {
      return summary;
    }
  }

  return fallback?.trim() || null;
}

export function parseResearchCommand(raw: string): { query: string; depth_mode: DepthMode } {
  const withoutPrefix = raw.replace(/^\/research\b/i, '').trim();

  if (withoutPrefix.startsWith('--max ')) {
    return {
      query: withoutPrefix.slice('--max '.length).trim(),
      depth_mode: 'max',
    };
  }

  if (withoutPrefix.toLowerCase().startsWith('max ')) {
    return {
      query: withoutPrefix.slice('max '.length).trim(),
      depth_mode: 'max',
    };
  }

  return {
    query: withoutPrefix,
    depth_mode: 'standard',
  };
}

function longestMarkerPrefixAtEnd(input: string, marker: string): number {
  const maxLength = Math.min(input.length, marker.length - 1);
  for (let length = maxLength; length > 0; length -= 1) {
    if (input.endsWith(marker.slice(0, length))) {
      return length;
    }
  }

  return 0;
}
