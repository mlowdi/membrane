import { describe, expect, it } from 'vitest';
import {
  OpenAIResponsesFormatter,
  OPENAI_RESPONSES_ITEMS_METADATA_KEY,
} from '../../src/formatters/openai-responses.js';

import { normalizeResponsesInput } from '../../src/providers/responses-input.js';

const options = {
  participantMode: 'multiuser' as const,
  assistantParticipant: 'Codex',
  systemPrompt: 'the recipe system prompt',
};

describe('OpenAIResponsesFormatter', () => {
  it('keeps tool-result images visual and ordered with the surrounding text', () => {
    const image = {
      type: 'image' as const,
      source: { type: 'base64' as const, mediaType: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==' },
    };
    const result = new OpenAIResponsesFormatter().buildMessages([
      { participant: 'Codex', content: [{ type: 'tool_use', id: 'vision', name: 'screenshot', input: {} }] },
      { participant: 'user', content: [{ type: 'tool_result', toolUseId: 'vision', content: [
        { type: 'text', text: 'before image' }, image,
        { type: 'text', text: 'after image' },
        { type: 'image', source: { type: 'url', url: 'https://example.test/screenshot.png' } },
      ] }] },
    ], options);
    expect(result.messages).toEqual([
      { type: 'function_call', call_id: 'vision', name: 'screenshot', arguments: '{}' },
      { type: 'function_call_output', call_id: 'vision', output: [
        { type: 'input_text', text: 'before image' },
        { type: 'input_image', image_url: `data:image/png;base64,${image.source.data}` },
        { type: 'input_text', text: 'after image' },
        { type: 'input_image', image_url: 'https://example.test/screenshot.png' },
      ] },
    ]);
  });

  it('normalizes maintenance-call image results without rewriting native replay items', () => {
    const replay = {
      type: 'function_call_output', call_id: 'old-vision', output: [
        { type: 'input_image', image_url: 'https://example.test/old.png' },
      ],
    };
    const result = normalizeResponsesInput([
      replay,
      { role: 'user', content: [
        { type: 'tool_result', toolUseId: 'new-vision', content: [
          { type: 'text', text: 'image from a tool' },
          { type: 'image', source: { type: 'base64', mediaType: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==' } },
        ] },
        { type: 'text', text: 'summarize this exchange' },
      ] },
    ]);
    expect(result).toEqual([
      replay,
      { type: 'function_call_output', call_id: 'new-vision', output: [
        { type: 'input_text', text: 'image from a tool' },
        { type: 'input_image', image_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==' },
      ] },
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'summarize this exchange' }] },
    ]);
  });
  it('replays native items exactly, then converts only the new tail', () => {
    const reasoning = {
      type: 'reasoning', id: 'rs_1', encrypted_content: 'opaque', summary: [],
    };
    const assistant = {
      type: 'message', id: 'msg_1', role: 'assistant', phase: 'final_answer',
      content: [{ type: 'output_text', text: 'old answer' }],
    };
    const formatter = new OpenAIResponsesFormatter();
    const result = formatter.buildMessages([
      {
        participant: 'Codex',
        content: [
          { type: 'redacted_thinking', data: 'opaque', rawItem: reasoning },
          { type: 'text', text: 'old answer', rawItem: assistant },
        ],
      },
      { participant: 'user', content: [{ type: 'text', text: 'new turn' }] },
    ], options);

    expect(result.messages).toEqual([
      reasoning,
      assistant,
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'new turn' }] },
    ]);
  });

  it('keeps the system prompt on turn 2+ of a fresh session (rawItem history is not an import)', () => {
    // Fresh session: turn 1 sent the system prompt as the top-level
    // `instructions` request field, never as an input item. The response's
    // blocks all carry rawItem, but that must NOT suppress the system prompt
    // on subsequent turns — the adapter is stateless, so nothing else
    // retains it.
    const formatter = new OpenAIResponsesFormatter();

    // Turn 1: no history at all.
    const turn1 = formatter.buildMessages([
      { participant: 'user', content: [{ type: 'text', text: 'hello' }] },
    ], options);
    expect(turn1.systemContent).toBe('the recipe system prompt');

    // Turn 2: history now contains the turn-1 response with rawItem blocks.
    const reasoning = { type: 'reasoning', id: 'rs_t1', encrypted_content: 'enc', summary: [] };
    const assistant = {
      type: 'message', id: 'msg_t1', role: 'assistant', phase: 'final_answer',
      content: [{ type: 'output_text', text: 'hi there' }],
    };
    const turn2 = formatter.buildMessages([
      { participant: 'user', content: [{ type: 'text', text: 'hello' }] },
      {
        participant: 'Codex',
        content: [
          { type: 'redacted_thinking', data: 'enc', rawItem: reasoning },
          { type: 'text', text: 'hi there', rawItem: assistant },
        ],
      },
      { participant: 'user', content: [{ type: 'text', text: 'and turn two?' }] },
    ], options);
    expect(turn2.systemContent).toBe('the recipe system prompt');
    expect(turn2.messages).toEqual([
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hello' }] },
      reasoning,
      assistant,
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'and turn two?' }] },
    ]);
  });

  it('suppresses the system prompt when history was imported via metadata items', () => {
    const developer = {
      type: 'message', role: 'developer',
      content: [{ type: 'input_text', text: 'imported rollout instructions' }],
    };
    const imported = [
      developer,
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'old turn' }] },
    ];
    const result = new OpenAIResponsesFormatter().buildMessages([
      {
        participant: 'user',
        content: [{ type: 'text', text: '(imported)' }],
        metadata: { [OPENAI_RESPONSES_ITEMS_METADATA_KEY]: imported },
      },
      { participant: 'user', content: [{ type: 'text', text: 'new turn' }] },
    ], options);

    expect(result.messages).toEqual([
      ...imported,
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'new turn' }] },
    ]);
    expect(result.systemContent).toBeUndefined();
  });

  it('suppresses the system prompt when the native prefix carries a developer item', () => {
    const developer = {
      type: 'message', id: 'dev_1', role: 'developer',
      content: [{ type: 'input_text', text: 'imported instructions' }],
    };
    const result = new OpenAIResponsesFormatter().buildMessages([
      {
        participant: 'user',
        content: [{ type: 'text', text: '', rawItem: developer }],
      },
      { participant: 'user', content: [{ type: 'text', text: 'new turn' }] },
    ], options);

    expect(result.messages).toEqual([
      developer,
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'new turn' }] },
    ]);
    expect(result.systemContent).toBeUndefined();
  });

  it('deduplicates a multi-part assistant item without losing its phase', () => {
    const item = {
      type: 'message', id: 'msg_multi', role: 'assistant', phase: 'commentary',
      content: [
        { type: 'output_text', text: 'a' },
        { type: 'output_text', text: 'b' },
      ],
    };
    const result = new OpenAIResponsesFormatter().buildMessages([{
      participant: 'Codex',
      content: [
        { type: 'text', text: 'a', rawItem: item },
        { type: 'text', text: 'b', rawItem: structuredClone(item) },
      ],
    }], options);

    expect(result.messages).toEqual([item]);
  });
});
