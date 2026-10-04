import { afterEach, describe, expect, it, vi } from 'vitest';
import { Membrane, NativeFormatter, OpenAIResponsesFormatter, AnthropicXmlFormatter,
  OpenAIResponsesAPIAdapter, GeminiAdapter, filterImageMessages, normalizeImageContent,
  projectResponsesItem, projectResponsesGeneratedImage, estimateImagePolicyContentTokens } from '../../src/index.js';
import { formatToolResultsForSplitTurn } from '../../src/utils/tool-parser.js';
import { normalizeResponsesInput } from '../../src/providers/responses-input.js';
import type { ContentBlock, GeneratedImageContent, ProviderAdapter, ProviderRequest, ProviderResponse, StreamCallbacks } from '../../src/types/index.js';

const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
const generated = (tokens = 731, preview = false): GeneratedImageContent => ({ type: 'generated_image',
  data: PNG, mimeType: 'image/png', tokenEstimate: tokens, isPreview: preview });
const text = (value: string): ContentBlock => ({ type: 'text', text: value });
const options = { participantMode: 'simple' as const, assistantParticipant: 'Claude', humanParticipant: 'User' };
function parts(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) return value.flatMap(parts);
  if (!value || typeof value !== 'object') return [];
  const record = value as Record<string, unknown>;
  return [record, ...Object.values(record).flatMap(parts)];
}
function images(value: unknown) { return parts(value).filter(part => part.type === 'image' || part.type === 'input_image'); }
function texts(value: unknown) { return parts(value).filter(part => typeof part.text === 'string').map(part => String(part.text)); }
class GeneratedProvider implements ProviderAdapter {
  readonly name = 'generated-output-fixture';
  constructor(readonly content: ContentBlock[]) {}
  supportsModel() { return true; }
  async complete(_request: ProviderRequest): Promise<ProviderResponse> {
    return { content: this.content, rawRequest: _request, stopReason: 'end_turn', usage: { inputTokens: 10, outputTokens: 5 }, raw: {} };
  }
  async stream(request: ProviderRequest, callbacks: StreamCallbacks): Promise<ProviderResponse> {
    this.content.forEach((block, index) => {
      callbacks.onContentBlock?.(index, block);
      if (block.type === 'text') callbacks.onChunk(block.text);
    });
    return this.complete(request);
  }
}
afterEach(() => vi.unstubAllGlobals());

describe('generated visuals use the canonical image boundary', () => {
  for (const Formatter of [NativeFormatter, OpenAIResponsesFormatter, AnthropicXmlFormatter]) {
    for (const participant of ['User', 'Claude']) {
      it(`${Formatter.name} retains original ${participant} generated visual and text siblings`, () => {
        const content = [text('before'), generated(0, true), text('between'), generated(), text('after')];
        const original = structuredClone(content);
        const built = new Formatter().buildMessages([{ participant, content }], options);
        expect(images(built.messages)).toHaveLength(2);
        const order = parts(built.messages).filter(part => part.type === 'image' || part.type === 'input_image' ||
          ['before', 'between', 'after'].includes(String(part.text))).map(part => typeof part.text === 'string' ? part.text : 'image');
        expect(order).toEqual(['before', 'image', 'between', 'image', 'after']);
        expect(texts(built.messages).join(' ')).toContain('before');
        expect(texts(built.messages).join(' ')).toContain('between');
        expect(texts(built.messages).join(' ')).toContain('after');
        expect(texts(built.messages).join(' ')).not.toContain(PNG);
        expect(content).toEqual(original);
        if (Formatter === OpenAIResponsesFormatter && participant === 'Claude') {
          const messages = built.messages as unknown as Array<{ role: string; content: Array<Record<string, unknown>> }>;
          expect(messages.map(message => message.role)).toEqual(['assistant', 'user', 'assistant', 'user', 'assistant']);
          expect(messages[0].content).toEqual([{ type: 'output_text', text: 'before' }]);
          expect(messages[2].content).toEqual([{ type: 'output_text', text: 'between' }]);
          expect(messages[1].content[0]).toEqual({ type: 'input_text',
            text: 'prior assistant-generated visual context; not user authorship or a new user request' });
        }
      });
    }
  }
  it('recursive Native/Responses/XML tool outputs stay visual and preserve intervening captions', () => {
    const content: ContentBlock[] = [text('before'), generated(), { type: 'tool_result', toolUseId: 'inner',
      content: [text('between'), generated(0), text('after')] }];
    for (const Formatter of [NativeFormatter, OpenAIResponsesFormatter, AnthropicXmlFormatter]) {
      const wire = new Formatter().buildMessages([
        { participant: 'Claude', content: [{ type: 'tool_use', id: 'outer', name: 'inspect', input: {} }] },
        { participant: 'User', content: [{ type: 'tool_result', toolUseId: 'outer', content }] },
      ], options).messages;
      expect(images(wire), Formatter.name).toHaveLength(2);
      expect(texts(wire).join(' ')).not.toContain(PNG);
      expect(texts(wire).join(' ')).toContain('between');
    }
    const split = formatToolResultsForSplitTurn([{ toolUseId: 'outer', content }]);
    expect(split.userContent.filter(block => block.type === 'image')).toHaveLength(2);
    expect(JSON.stringify(split)).toContain('between');
  });
  it('declared MIME floor and supported-label correction match canonical images', () => {
    for (const mimeType of [undefined, 'image/svg+xml', null, 42]) {
      const block = { type: 'generated_image', data: PNG, mimeType } as unknown as GeneratedImageContent;
      expect(normalizeImageContent(block as unknown as Record<string, unknown>)).toMatchObject({ type: 'text' });
      for (const Formatter of [NativeFormatter, OpenAIResponsesFormatter, AnthropicXmlFormatter]) {
        const wire = new Formatter().buildMessages([{ participant: 'User', content: [text('before'), block, text('after')] }], options).messages;
        expect(images(wire)).toHaveLength(0);
        expect(JSON.stringify(wire)).not.toContain(PNG);
      }
    }
    expect(normalizeImageContent({ ...generated(), mimeType: 'image/webp' })).toMatchObject({ type: 'generated_image', mimeType: 'image/png' });
  });
  it('generated and canonical visuals share recursive count/bytes/depth and explicit-zero pricing', () => {
    const content: ContentBlock[] = [{ type: 'image', source: { type: 'base64', data: PNG, mediaType: 'image/png' } },
      { type: 'tool_result', toolUseId: 'nested', content: [generated(0), generated(731)] }];
    expect(estimateImagePolicyContentTokens(content)).toBe(2331);
    const source = [{ participant: 'User', content }];
    for (const policy of [{ maxLiveImages: 1, maxLiveImageBytes: 0 }, { maxLiveImages: 0, maxLiveImageBytes: PNG.length }]) {
      const filtered = filterImageMessages(source, policy);
      expect(parts(filtered).filter(part => part.type === 'generated_image')).toHaveLength(1);
      expect(parts(filtered).filter(part => part.type === 'image')).toHaveLength(0);
      expect(texts(filtered).join(' ')).not.toContain(PNG);
    }
    const filtered = filterImageMessages([...source, { participant: 'User', content: [text('x'.repeat(100))] }],
      { imageStripDepthTokens: 20, maxLiveImageBytes: 0 });
    expect(parts(filtered).filter(part => part.type === 'generated_image' || part.type === 'image')).toHaveLength(0);
    expect(parts(source).filter(part => part.type === 'generated_image')).toHaveLength(2);
  });
  it('raw native generation is replayed once, with no companion or filtered-payload resurrection', () => {
    const item = { type: 'image_generation_call', id: 'generation', status: 'completed', result: PNG, output_format: 'png' };
    const source = [{ participant: 'Claude', content: projectResponsesItem(item) }];
    expect(new OpenAIResponsesFormatter().buildMessages(source, options).messages).toEqual([item]);
    expect(images(new NativeFormatter().buildMessages(source, options).messages)).toHaveLength(1);
    const dropped = filterImageMessages(source, { maxLiveImageBytes: 1 });
    expect(JSON.stringify(new OpenAIResponsesFormatter().buildMessages(dropped, options).messages)).not.toContain(PNG);
    expect(source[0].content[0].rawItem).toBe(item);
  });
  it('falsifier: normalized transport replays authoritative native generation once without a companion', () => {
    const rawItem = { type: 'image_generation_call', id: 'fallback-native', status: 'completed', result: PNG, output_format: 'png' };
    const block = { ...generated(), rawItem };
    const source = [{ role: 'assistant', content: [text('before'), block, block, text('after')] }];
    const wire = normalizeResponsesInput(source);
    expect(wire).toEqual([
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'before' }] },
      rawItem,
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'after' }] },
    ]);
    expect(images(wire)).toHaveLength(0);
    expect(source[0].content[1]).toBe(block);
  });
  it('standalone normalized assistant transport uses the explicit-origin companion, not assistant input_text', () => {
    const wire = normalizeResponsesInput([{ role: 'assistant', content: [text('before'), generated(), text('after')] }]);
    expect(wire.map(item => item.role)).toEqual(['assistant', 'user', 'assistant']);
    expect(wire[0].content).toEqual([{ type: 'output_text', text: 'before' }]);
    expect(texts(wire).join(' ')).toContain('not user authorship or a new user request');
    expect(images(wire)).toHaveLength(1);
  });
});

for (const mode of ['complete', 'stream', 'yielding'] as const) {
  for (const toolMode of ['native', 'xml'] as const) {
    it(`${mode}/${toolMode} preserves provider generated preview/final order, estimates and opaque metadata`, async () => {
      const rawItem = { type: 'foreign-native-image', nativeBytes: PNG };
      const preview = { ...generated(0, true), rawItem, producerLabel: 'unchanged' };
      const final = { ...generated(731, false), rawItem, producerLabel: 'final' };
      const content = [text('before'), preview, text('between'), final, text('after')];
      const membrane = new Membrane(new GeneratedProvider(content), { formatter: toolMode === 'native' ? new NativeFormatter() : new AnthropicXmlFormatter() });
      const request = { messages: [{ participant: 'User', content: [text('inspect')] }], toolMode,
        config: { model: 'fixture', maxTokens: 100 } };
      const response = mode === 'complete' ? await membrane.complete(request)
        : mode === 'stream' ? await membrane.stream(request)
        : await (async () => {
          const stream = membrane.streamYielding(request);
          for await (const event of stream) if (event.type === 'complete') return event.response;
          throw new Error('no final response');
        })();
      expect(response.content.map(block => block.type)).toEqual(content.map(block => block.type));
      expect(response.content.filter(block => block.type === 'generated_image')).toEqual([preview, final]);
      expect(response.rawAssistantText).not.toContain(PNG);
    });
  }
}

it('real Responses output converter projects only grounded available native generation and retains opaque alternatives', async () => {
  const opaqueBytes = Buffer.from('not a recognizable raster').toString('base64');
  const available = [
    { type: 'image_generation_call', id: 'format', status: 'completed', result: PNG, output_format: 'webp' },
    { type: 'image_generation_call', id: 'signature', status: 'completed', result: PNG },
  ];
  const unavailable = [
    ...['in_progress', 'generating', 'failed', 'unknown'].map(status => ({ type: 'image_generation_call', status, result: PNG })),
    ...[null, undefined, '', 42, 'invalid!', opaqueBytes].map(result => ({ type: 'image_generation_call', status: 'completed', result })),
    { type: 'image_generation_call', status: 'completed', result: PNG, output_format: 'svg' },
    { type: 'unknown-native-kind', status: 'completed', result: PNG },
  ];
  const output = [...available, ...unavailable];
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ model: 'fixture', status: 'completed', output,
    usage: { input_tokens: 10, output_tokens: 5 } }), { status: 200, headers: { 'Content-Type': 'application/json' } })));
  const response = await new OpenAIResponsesAPIAdapter({ apiKey: 'fixture' }).complete({ model: 'fixture', maxTokens: 100, messages: [] });
  expect(response.outputItems).toEqual(output);
  const normalized = response.content as Array<Record<string, unknown>>;
  expect(normalized.slice(0, 2).map(block => block.type)).toEqual(['generated_image', 'generated_image']);
  expect(normalized[0]).toMatchObject({ mimeType: 'image/png', itemId: 'format', outputIndex: 0, rawItem: available[0] });
  expect(normalized.slice(2).every(block => block.type === 'openai_response_item')).toBe(true);
  for (const item of unavailable) expect(projectResponsesGeneratedImage(item)).toBeUndefined();
});

it('actual Gemini producer preserves alternating image/text output and native image part testimony', async () => {
  const imagePart = { inlineData: { data: PNG, mimeType: 'image/png' } };
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ candidates: [{ finishReason: 'STOP',
    content: { parts: [{ text: 'before' }, imagePart, { text: 'after' }] } }], usageMetadata: {} }),
    { status: 200, headers: { 'Content-Type': 'application/json' } })));
  const response = await new GeminiAdapter({ apiKey: 'fixture' }).complete({ model: 'gemini-image', maxTokens: 100, messages: [] });
  expect(response.content).toEqual([text('before'), { type: 'generated_image', data: PNG, mimeType: 'image/png', rawItem: imagePart }, text('after')]);
});
