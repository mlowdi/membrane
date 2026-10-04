import { describe, it, expect } from 'vitest';
import { Membrane } from '../../src/membrane.js';
import { OpenAIResponsesFormatter } from '../../src/formatters/openai-responses.js';
import { NativeFormatter } from '../../src/formatters/native.js';
import { filterImageMessages, projectResponsesItem, normalizeImageContent } from '../../src/utils/image-policy.js';
import type { ContentBlock, NormalizedRequest, ProviderAdapter, ProviderRequest, ProviderResponse,
  StreamCallbacks, ToolResult } from '../../src/types/index.js';

const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
const image = (): ContentBlock => ({ type: 'image', source: { type: 'base64', mediaType: 'image/png', data: PNG } });
const url = (label: string): ContentBlock => ({ type: 'image', source: { type: 'url', url: `https://example.test/${label}.png` } });
const text = (value: string): ContentBlock => ({ type: 'text', text: value });

class Rounds implements ProviderAdapter {
  readonly name = 'image-policy-fixture';
  readonly frames: ProviderRequest[] = [];
  constructor(private count: number) {}
  supportsModel(): boolean { return true; }
  async complete(request: ProviderRequest): Promise<ProviderResponse> {
    this.frames.push(request);
    return this.response(this.frames.length - 1);
  }
  async stream(request: ProviderRequest, callbacks: StreamCallbacks): Promise<ProviderResponse> {
    this.frames.push(request);
    const response = this.response(this.frames.length - 1);
    response.content.forEach((block, index) => {
      callbacks.onContentBlock?.(index, block);
      if (block.type === 'text' && typeof block.text === 'string') callbacks.onChunk(block.text);
    });
    return response;
  }
  private response(index: number): ProviderResponse {
    const content = index < this.count
      ? [{ type: 'tool_use', id: `round-${index}`, name: 'inspect', input: {} }]
      : [{ type: 'text', text: 'finished' }];
    return { content, stopReason: index < this.count ? 'tool_use' : 'end_turn',
      usage: { inputTokens: 20, outputTokens: 5 }, raw: {} };
  }
}

function parts(request: ProviderRequest): Array<Record<string, unknown>> {
  const result: Array<Record<string, unknown>> = [];
  const walk = (value: unknown): void => {
    if (Array.isArray(value)) { value.forEach(walk); return; }
    if (!value || typeof value !== 'object') return;
    if ('type' in value && (value.type === 'input_image' || value.type === 'input_text')) {
      result.push(value as Record<string, unknown>);
    }
    Object.values(value).forEach(walk);
  };
  walk(request.messages);
  return result;
}
const visualUrls = (request: ProviderRequest) => parts(request).filter(p => p.type === 'input_image').map(p => p.image_url);
const promptText = (request: ProviderRequest) => parts(request).filter(p => p.type === 'input_text').map(p => p.text).join('\n');

function request(policy: NormalizedRequest['liveImagePolicy']): NormalizedRequest {
  return { messages: [{ participant: 'User', content: [text('original'), url('old')] }],
    tools: [{ name: 'inspect', description: 'inspect', inputSchema: { type: 'object', properties: {} } }],
    toolMode: 'native', liveImagePolicy: policy, config: { model: 'fixture', maxTokens: 100 } };
}

async function run(mode: 'yielding' | 'stream', adapter: Rounds, input: NormalizedRequest, results: ContentBlock[][]) {
  const membrane = new Membrane(adapter, { formatter: new OpenAIResponsesFormatter() });
  if (mode === 'stream') {
    await membrane.stream(input, { onToolCalls: async calls => calls.map(call => ({
      toolUseId: call.id, content: results[Number(call.id.split('-')[1])],
    })) });
  } else {
    const stream = membrane.streamYielding(input);
    for await (const event of stream) {
      if (event.type === 'tool-calls') stream.provideToolResults(event.calls.map(call => ({
        toolUseId: call.id, content: results[Number(call.id.split('-')[1])],
      })));
    }
  }
}

describe('live image policy at the actual native request boundary', () => {
  for (const mode of ['yielding', 'stream'] as const) {
    it(`${mode}: bounds retained and newly returned recursive images across rounds`, async () => {
      const adapter = new Rounds(2);
      const first = [text('before'), image(), text('between'), url('one'), text('after')];
      const second: ContentBlock[] = [{ type: 'tool_result', toolUseId: 'recalled',
        content: [text('nested-before'), url('two'), url('three'), text('nested-after')] }];
      const input = request({ maxLiveImages: 2, maxLiveImageBytes: PNG.length * 2, imageStripDepthTokens: 0 });
      await run(mode, adapter, input, [first, second]);
      expect(adapter.frames).toHaveLength(3);
      expect(visualUrls(adapter.frames[1])).toEqual([`data:image/png;base64,${PNG}`, 'https://example.test/one.png']);
      expect(visualUrls(adapter.frames[2])).toEqual(['https://example.test/two.png', 'https://example.test/three.png']);
      const wire = adapter.frames[1].messages as Array<Record<string, unknown>>;
      const output = wire.find(item => item.type === 'function_call_output' && item.call_id === 'round-0');
      expect(output?.output).toEqual([
        { type: 'input_text', text: 'before' }, { type: 'input_image', image_url: `data:image/png;base64,${PNG}` },
        { type: 'input_text', text: 'between' }, { type: 'input_image', image_url: 'https://example.test/one.png' },
        { type: 'input_text', text: 'after' },
      ]);
      expect((adapter.frames[2].messages as Array<Record<string, unknown>>)
        .filter(item => item.type === 'function_call_output').map(item => item.call_id)).toEqual(['round-0', 'round-1']);
      expect(promptText(adapter.frames[2])).not.toContain(PNG);
      expect(promptText(adapter.frames[2]).length).toBeGreaterThan(0);
      expect(first[1]).toEqual(image());
      expect(second[0]).toMatchObject({ content: [text('nested-before'), url('two'), url('three'), text('nested-after')] });
      expect(input.messages[0].content[1]).toEqual(url('old'));
    });

    it(`${mode}: enforces bytes independently of count and leaves an explicit image slot`, async () => {
      const adapter = new Rounds(1);
      await run(mode, adapter, request({ maxLiveImages: 0, maxLiveImageBytes: PNG.length, imageStripDepthTokens: 0 }),
        [[image(), image()]]);
      expect(visualUrls(adapter.frames[1]).filter(value => String(value).startsWith('data:'))).toHaveLength(1);
      const output = (adapter.frames[1].messages as Array<Record<string, unknown>>).find(item => item.type === 'function_call_output');
      const outputParts = output?.output as Array<Record<string, unknown>>;
      expect(outputParts.map(part => part.type)).toEqual(['input_text', 'input_image']);
      expect(String(outputParts[0].text).length).toBeGreaterThan(0);
      expect(promptText(adapter.frames[1])).not.toContain(PNG);
    });

    it(`${mode}: explicit zero dimensions keep every permitted image`, async () => {
      const adapter = new Rounds(1);
      await run(mode, adapter, request({ maxLiveImages: 0, maxLiveImageBytes: 0, imageStripDepthTokens: 0 }),
        [[image(), url('one'), url('two')]]);
      expect(visualUrls(adapter.frames[1])).toHaveLength(4);
    });
  }

  it('complete filters imported native images without changing IDs, phases, or signed carriers in the archive', async () => {
    const reasoning = { type: 'reasoning', id: 'reasoning-id', encrypted_content: 'signed-ciphertext', summary: [] };
    const tool = { type: 'function_call_output', id: 'output-id', call_id: 'call-id',
      output: [{ type: 'input_text', text: 'before' }, { type: 'input_image', image_url: `data:image/png;base64,${PNG}` },
        { type: 'input_text', text: 'after' }] };
    const user = { type: 'message', role: 'user', id: 'user-id',
      content: [{ type: 'input_image', image_url: 'https://example.test/new.png' }] };
    const assistant = { type: 'message', role: 'assistant', id: 'assistant-id', phase: 'commentary',
      content: [{ type: 'output_text', text: 'inspect' }] };
    const native = [reasoning, assistant, { type: 'function_call', call_id: 'call-id', name: 'inspect', arguments: '{}' }, tool, user];
    const messages = [{ participant: 'User', content: native.flatMap(projectResponsesItem), metadata: { openaiResponsesItems: native } }];
    const before = JSON.stringify(messages);
    const adapter = new Rounds(0);
    await new Membrane(adapter, { formatter: new OpenAIResponsesFormatter() }).complete({ ...request({ maxLiveImages: 1,
      maxLiveImageBytes: 0, imageStripDepthTokens: 0 }), messages });
    expect(visualUrls(adapter.frames[0])).toEqual(['https://example.test/new.png']);
    const wire = adapter.frames[0].messages as Array<Record<string, unknown>>;
    expect(wire.find(item => item.type === 'reasoning')).toEqual(reasoning);
    expect(wire.find(item => item.id === 'assistant-id')).toEqual(assistant);
    expect(wire.find(item => item.type === 'function_call_output')).toMatchObject({ id: 'output-id', call_id: 'call-id' });
    expect(promptText(adapter.frames[0])).not.toContain(PNG);
    expect(JSON.stringify(messages)).toBe(before);
    const unchanged = new OpenAIResponsesFormatter().buildMessages(messages, { participantMode: 'multiuser', assistantParticipant: 'Codex' });
    expect(unchanged.messages).toEqual(native);
  });

  it('the same imported output remains visual on an auxiliary native formatter', () => {
    const output = { type: 'function_call_output', call_id: 'vision', output: [
      { type: 'input_text', text: 'before' }, { type: 'input_image', image_url: `data:image/png;base64,${PNG}` },
      { type: 'input_text', text: 'after' },
    ] };
    const built = new NativeFormatter().buildMessages([
      { participant: 'Codex', content: projectResponsesItem({ type: 'function_call', call_id: 'vision', name: 'inspect', arguments: '{}' }) },
      { participant: 'User', content: projectResponsesItem(output) },
    ], { participantMode: 'multiuser', assistantParticipant: 'Codex' });
    const tool = built.messages.flatMap(message => message.content as Array<Record<string, unknown>>)
      .find(block => block.type === 'tool_result');
    expect(tool?.tool_use_id).toBe('vision');
    expect(tool?.content).toEqual([{ type: 'text', text: 'before' },
      { type: 'image', source: { type: 'base64', media_type: 'image/png', data: PNG } }, { type: 'text', text: 'after' }]);
  });

  it('depth pricing includes recursive images rather than their encoded strings', () => {
    const messages = [{ participant: 'User', content: [image()] },
      { participant: 'User', content: [{ type: 'tool_result', toolUseId: 'nested', content: [image()] } as ContentBlock] }];
    const bounded = filterImageMessages(messages, { maxLiveImages: 0, maxLiveImageBytes: 0, imageStripDepthTokens: 1700 });
    expect(bounded[0].content[0].type).toBe('text');
    expect(bounded[1].content[0]).toMatchObject({ type: 'tool_result', content: [{ type: 'image' }] });
  });

  it('direct formatter calls keep unsupported MIME and malformed base64 unavailable without rewriting input', () => {
    const invalid: ContentBlock[] = [
      { type: 'image', source: { type: 'base64', mediaType: 'image/svg+xml', data: PNG } },
      { type: 'image', source: { type: 'base64', mediaType: 'image/png', data: '%%%%' } },
      { type: 'image', source: { type: 'base64', mediaType: 'image/png', data: `${PNG}=` } },
      { type: 'image', source: { type: 'url', url: `data:image/svg+xml;base64,${PNG}` } },
    ];
    for (const formatter of [new OpenAIResponsesFormatter(), new NativeFormatter()]) {
      for (const block of invalid) {
        const messages = [{ participant: 'User', content: [text('before'), block, text('after')] }];
        const original = JSON.stringify(messages);
        const built = formatter.buildMessages(messages, {
          participantMode: 'multiuser', assistantParticipant: 'Codex', promptCaching: false,
        });
        const serialized = JSON.stringify(built.messages);
        expect(serialized).not.toContain('input_image');
        expect(serialized).not.toContain('"type":"image"');
        expect(serialized).not.toContain(PNG);
        expect(serialized).not.toContain('%%%%');
        expect(serialized).toContain('before');
        expect(serialized).toContain('after');
        expect(serialized.length).toBeLessThan(500);
        expect(JSON.stringify(messages)).toBe(original);
      }
    }
  });

  it('decodable PNGs with nonzero unused pad bits stay visual and their native carrier stays exact', () => {
    const decodable = `${PNG.slice(0, -3)}h==`;
    expect(Buffer.from(decodable, 'base64')).toEqual(Buffer.from(PNG, 'base64'));
    const item = { type: 'message', id: 'pad-bit-user', role: 'user', content: [
      { type: 'input_image', image_url: `data:image/png;base64,${decodable}` },
    ] };
    const content = projectResponsesItem(item);
    expect(content[0].type).toBe('image');
    const replay = new OpenAIResponsesFormatter().buildMessages([{ participant: 'User', content }], {
      participantMode: 'multiuser', assistantParticipant: 'Codex', promptCaching: false,
    });
    expect(replay.messages).toEqual([item]);
    const auxiliary = new NativeFormatter().buildMessages([{ participant: 'User', content }], {
      participantMode: 'multiuser', assistantParticipant: 'Codex', promptCaching: false,
    });
    expect(auxiliary.messages[0].content).toEqual([{ type: 'image', source: {
      type: 'base64', media_type: 'image/png', data: decodable,
    } }]);
  });

  it('invalid MCP image shapes cannot turn their bytes into ordinary prompt text', () => {
    for (const invalid of [{ type: 'image', data: PNG }, { type: 'image', data: PNG, mimeType: 'image/svg+xml' },
      { type: 'image', source: { type: 'unknown', data: PNG, mediaType: 'image/png' } },
      { type: 'image', source: { type: 'url', url: `data:application/octet-stream;base64,${PNG}` } }]) {
      const result = normalizeImageContent(invalid);
      expect(result.type).toBe('text');
      expect(JSON.stringify(result)).not.toContain(PNG);
      expect(JSON.stringify(result).length).toBeLessThan(200);
    }
  });
});
