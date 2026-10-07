import { describe, expect, it } from 'vitest';
import { OpenAIResponsesFormatter } from '../../src/formatters/openai-responses.js';
import { normalizeResponsesInput, responsesToolResultOutput } from '../../src/providers/responses-input.js';
import { IMAGE_UNAVAILABLE_TEXT } from '../../src/utils/image-policy.js';
import type { ProviderRequest, ToolResult } from '../../src/types/index.js';

const options = {
  participantMode: 'multiuser' as const,
  assistantParticipant: 'Agent',
  systemPrompt: 'sys',
};

const png = { type: 'base64' as const, mediaType: 'image/png', data: 'iVBORw0KGgo=' };

describe('tool results carrying images (Responses)', () => {
  it('formatter emits native input_text/input_image parts, not base64-as-text', () => {
    const formatter = new OpenAIResponsesFormatter();
    const result = formatter.buildMessages([
      { participant: 'Agent', content: [{ type: 'tool_use', id: 'call_1', name: 'snapshot', input: {} }] },
      {
        participant: 'user',
        content: [{
          type: 'tool_result', toolUseId: 'call_1',
          content: [{ type: 'text', text: 'first-person view' }, { type: 'image', source: png }],
        }],
      },
    ] as any, options);

    const output = (result.messages as any[]).find((m) => m.type === 'function_call_output');
    expect(output).toEqual({
      type: 'function_call_output',
      call_id: 'call_1',
      output: [
        { type: 'input_text', text: 'first-person view' },
        { type: 'input_image', image_url: 'data:image/png;base64,iVBORw0KGgo=' },
      ],
    });
  });

  it('formatter emits typed text parts for normalized image-free array content', () => {
    const formatter = new OpenAIResponsesFormatter();
    const content = [{ type: 'text', text: 'just text' }];
    const result = formatter.buildMessages([
      { participant: 'Agent', content: [{ type: 'tool_use', id: 'call_2', name: 'look', input: {} }] },
      { participant: 'user', content: [{ type: 'tool_result', toolUseId: 'call_2', content }] },
    ] as any, options);
    const output = (result.messages as any[]).find((m) => m.type === 'function_call_output');
    expect(output.output).toEqual([{ type: 'input_text', text: 'just text' }]);
  });

  it('subscription normalizer converts normalized tool_result images the same way', () => {
    const items = normalizeResponsesInput([
      {
        role: 'user',
        content: [{
          type: 'tool_result', toolUseId: 'call_3',
          content: [{ type: 'image', source: { type: 'url', url: 'https://example.com/a.png' } }],
        }],
      },
    ] as any);
    expect(items).toEqual([{
      type: 'function_call_output',
      call_id: 'call_3',
      output: [{ type: 'input_image', image_url: 'https://example.com/a.png' }],
    }]);
  });

  it('keeps direct formatter tool-result admission MIME-required', () => {
    const content = [
      { type: 'image', source: { ...png, mediaType: 'image/svg+xml' } },
      { type: 'image', source: { type: 'base64', data: png.data } },
    ];
    const original = structuredClone(content);
    // Deliberately malformed normalized blocks exercise the runtime admission boundary.
    const result = { toolUseId: 'strict', content } as unknown as ToolResult;
    expect(JSON.parse(new OpenAIResponsesFormatter().formatToolResults([result]))).toEqual([{
      type: 'function_call_output', call_id: 'strict',
      output: Array.from({ length: 2 }, () => ({ type: 'input_text', text: IMAGE_UNAVAILABLE_TEXT })),
    }]);
    expect(content).toEqual(original);
  });

  it('subscription transport sniffs nested tool-result images without altering the source', () => {
    const content = [{ type: 'tool_result', content: [
      { type: 'image', source: { ...png, mediaType: 'image/svg+xml' } },
      { type: 'image', source: { type: 'base64', data: png.data } },
    ] }];
    const original = structuredClone(content);
    const messages = [{ type: 'tool_result', toolUseId: 'transport', content }] as unknown as ProviderRequest['messages'];
    expect(normalizeResponsesInput(messages)).toEqual([{
      type: 'function_call_output', call_id: 'transport',
      output: Array.from({ length: 2 }, () => ({ type: 'input_image', image_url: 'data:image/png;base64,' + png.data })),
    }]);
    expect(content).toEqual(original);
  });

  it('helper preserves strings, projects text arrays and bounds unusable image sources', () => {
    expect(responsesToolResultOutput('plain')).toBe('plain');
    expect(responsesToolResultOutput([{ type: 'text', text: 'x' }])).toEqual([{ type: 'input_text', text: 'x' }]);
    expect(responsesToolResultOutput([
      { type: 'image', source: { type: 'file', path: '/tmp/x.png' } },
      { type: 'image', source: png },
      { type: 'other', value: 1 },
    ])).toEqual([
      { type: 'input_text', text: IMAGE_UNAVAILABLE_TEXT },
      { type: 'input_image', image_url: 'data:image/png;base64,iVBORw0KGgo=' },
      { type: 'input_text', text: '{"type":"other","value":1}' },
    ]);
  });

  it('helper keeps nested result seams and corrects admitted image signatures', () => {
    const content = [
      { type: 'text', text: 'before' },
      { type: 'tool_result', toolUseId: 'nested', content: [
        { type: 'image', source: { ...png, mediaType: 'image/webp' } },
        { type: 'text', text: 'between' },
        { type: 'tool_result', toolUseId: 'deep', content: [
          { type: 'input_image', image_url: 'data:image/webp;base64,iVBORw0KGgo=' },
        ] },
      ] },
      { type: 'text', text: 'after' },
    ];
    const original = structuredClone(content);
    expect(responsesToolResultOutput(content)).toEqual([
      { type: 'input_text', text: 'before' },
      { type: 'input_image', image_url: 'data:image/png;base64,iVBORw0KGgo=' },
      { type: 'input_text', text: 'between' },
      { type: 'input_image', image_url: 'data:image/png;base64,iVBORw0KGgo=' },
      { type: 'input_text', text: 'after' },
    ]);
    expect(content).toEqual(original);
  });
});
