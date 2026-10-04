/**
 * Unit tests for context-management primitives: createInitialState,
 * defaultTokenEstimator, DEFAULT_CONTEXT_CONFIG.
 *
 * Converted from the legacy tsx script test/context.test.ts (pre-vitest
 * layout, never ran in CI). That script's remaining sections asserted
 * language behavior (spread copies, JSON round-trips) against local test
 * helpers rather than library exports, so they were retired instead of
 * ported.
 */

import { describe, it, expect } from 'vitest';
import {
  createInitialState,
  defaultTokenEstimator,
  DEFAULT_CONTEXT_CONFIG,
} from '../../src/context/index.js';
import type { NormalizedMessage } from '../../src/types/index.js';

function textMessage(text: string): NormalizedMessage {
  return { participant: 'User', content: [{ type: 'text', text }] };
}

describe('createInitialState', () => {
  it('starts empty, unrolled, and out of grace', () => {
    const state = createInitialState();
    expect(state.cacheMarkers).toEqual([]);
    expect(state.windowMessageIds).toEqual([]);
    expect(state.messagesSinceRoll).toBe(0);
    expect(state.inGracePeriod).toBe(false);
  });
});

describe('defaultTokenEstimator', () => {
  it('estimates ~4 chars per token, rounding up', () => {
    expect(defaultTokenEstimator(textMessage('Hello world'))).toBe(3); // 11 chars
    expect(defaultTokenEstimator(textMessage('x'.repeat(400)))).toBe(100);
  });

  it('falsifier: generated and nested visuals honor explicit estimates without binary or opaque JSON', () => {
    const opaque = { toJSON() { throw new Error('opaque carrier must not be serialized'); } };
    const generated = { type: 'generated_image' as const, data: 'A'.repeat(1_000_000), mimeType: 'image/png', tokenEstimate: 731, rawItem: opaque };
    const estimate = (content: NormalizedMessage['content']) => defaultTokenEstimator({ participant: 'User', content });
    expect(estimate([generated])).toBe(731);
    expect(estimate([{ ...generated, tokenEstimate: 0 }])).toBe(0);
    expect(estimate([{ type: 'image', source: { type: 'base64', data: 'AA==', mediaType: 'image/png' } }])).toBe(1500);
    expect(estimate([{ ...generated, tokenEstimate: undefined }])).toBe(1500);
    expect(estimate([{ type: 'text', text: 'a' }, { type: 'tool_result', toolUseId: 'outer', content: [
      { type: 'text', text: 'b' }, { type: 'tool_result', toolUseId: 'inner', content: [generated] }], rawItem: opaque }])).toBe(732);
    const textOnly = [{ type: 'text' as const, text: 'a' }, { type: 'text' as const, text: 'b' }];
    expect(estimate([{ type: 'tool_result', toolUseId: 'text', content: textOnly }])).toBe(Math.ceil(JSON.stringify(textOnly).length / 4));
    expect(estimate([{ type: 'tool_result', toolUseId: 'text', content: 'abcde' }])).toBe(2);
  });
});

describe('DEFAULT_CONTEXT_CONFIG', () => {
  it('carries the documented rolling/cache defaults', () => {
    expect(DEFAULT_CONTEXT_CONFIG.rolling.threshold).toBe(50);
    expect(DEFAULT_CONTEXT_CONFIG.rolling.buffer).toBe(20);
    expect(DEFAULT_CONTEXT_CONFIG.rolling.unit).toBe('messages');
    expect(DEFAULT_CONTEXT_CONFIG.cache?.enabled).toBe(true);
    expect(DEFAULT_CONTEXT_CONFIG.cache?.points).toBe(1);
  });
});
