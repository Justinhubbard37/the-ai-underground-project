import { z } from 'zod';

export const executeLocalGemmaPayloadSchema = z.object({
  prompt: z.string().trim().min(1, 'Prompt is required.'),
  scalpel_context: z.string().trim().min(1).nullable(),
});

export const dispatchDeepResearchPayloadSchema = z.object({
  query: z.string().trim().min(1, 'Research query is required.'),
  depth_mode: z.enum(['standard', 'max']),
});

export const localGemmaUpdateSchema = z.object({
  run_id: z.string().min(1),
  delta: z.string(),
  done: z.boolean(),
  error: z.string().nullable(),
});

export const researchUpdateSchema = z.object({
  task_id: z.string().min(1),
  query: z.string(),
  status: z.enum(['queued', 'running', 'completed', 'failed']),
  active_step: z.string(),
  thought_summary: z.string().nullable(),
  steps: z.array(
    z.object({
      id: z.string().optional(),
      title: z.string().optional(),
      active_step: z.string().optional(),
      thought_summary: z.string().optional(),
      status: z.enum(['queued', 'running', 'completed', 'failed']).optional(),
    }),
  ),
  artifact: z.string().nullable(),
  error: z.string().nullable(),
});
