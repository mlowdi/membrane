/**
 * Split-turn injection for images in tool results (prefill mode):
 * hasImageInToolResults and formatToolResultsForSplitTurn.
 * Converted from the legacy tsx script test/tool-result-images.test.ts
 * (pre-vitest layout, never ran in CI). formatToolResults itself is covered
 * in tool-parser.test.ts; only its image-placeholder mode is pinned here.
 */

import { describe, it, expect } from 'vitest';
import {
  hasImageInToolResults,
  formatToolResultsForSplitTurn,
  formatToolResults,
} from '../../src/utils/tool-parser.js';
import type { ToolResult, ToolResultContentBlock } from '../../src/types/index.js';

// 1x1 red pixel PNG
const sampleImageData =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==';

function createTextResult(toolUseId: string, text: string): ToolResult {
  return { toolUseId, toolName: `name_${toolUseId}`, content: text };
}

function createImageResult(toolUseId: string, text: string): ToolResult {
  const content: ToolResultContentBlock[] = [
    { type: 'text', text },
    { type: 'image', source: { type: 'base64', data: sampleImageData, mediaType: 'image/png' } },
  ];
  return { toolUseId, toolName: `name_${toolUseId}`, content };
}

function createMultiImageResult(toolUseId: string, text: string, imageCount: number): ToolResult {
  const content: ToolResultContentBlock[] = [{ type: 'text', text }];
  for (let i = 0; i < imageCount; i++) {
    content.push({
      type: 'image',
      source: { type: 'base64', data: sampleImageData, mediaType: 'image/png' },
    });
  }
  return { toolUseId, toolName: `name_${toolUseId}`, content };
}

describe('hasImageInToolResults', () => {
  it('detects images across result shapes', () => {
    expect(hasImageInToolResults([createTextResult('tool_1', 'Hello world')])).toBe(false);
    expect(hasImageInToolResults([createImageResult('tool_1', 'Image result')])).toBe(true);
    expect(
      hasImageInToolResults([
        createTextResult('tool_1', 'Text result'),
        createImageResult('tool_2', 'Image result'),
      ]),
    ).toBe(true);
    expect(hasImageInToolResults([{ toolUseId: 'tool_1', content: [] }])).toBe(false);
  });
});

describe('formatToolResults image placeholder mode', () => {
  it('renders images as bracketed placeholders inside the XML', () => {
    const xml = formatToolResults([createImageResult('tool_1', 'Screenshot taken')]);
    expect(xml).toContain('<function_results>');
    expect(xml).toContain('Screenshot taken');
    expect(xml).toContain('[Image: image/png');
    expect(xml).toContain('</function_results>');
  });
});

describe('formatToolResultsForSplitTurn', () => {
  it('preserves both URL images and intervening text across result boundaries', () => {
    const first = { type: 'image' as const, source: { type: 'url' as const, url: 'https://example.test/first.png' } };
    const second = { type: 'image' as const, source: { type: 'url' as const, url: 'https://example.test/second.png' } };
    const split = formatToolResultsForSplitTurn([
      { toolUseId: 'first', content: [{ type: 'text', text: 'before first' }, first, { type: 'text', text: 'after first' }] },
      { toolUseId: 'second', content: [{ type: 'text', text: 'before second' }, second, { type: 'text', text: 'after second' }] },
    ]);
    expect(split.userContent.filter(block => block.type === 'image')).toEqual([first, second]);
    const between = split.userContent.filter(block => block.type === 'text').map(block => block.text).join('');
    expect(between.indexOf('after first')).toBeGreaterThanOrEqual(0);
    expect(between.indexOf('before second')).toBeGreaterThan(between.indexOf('after first'));
    expect(split.beforeImageXml).toContain('before first');
    expect(split.afterImageXml).toContain('after second');
  });

  it('detects and preserves recursively nested URL and base64 image leaves without encoded XML text', () => {
    const linked = { type: 'image' as const, source: { type: 'url' as const, url: 'https://example.test/nested.png' } };
    const result: ToolResult = { toolUseId: 'outer', content: [{ type: 'tool_result', toolUseId: 'nested', content: [
      { type: 'text', text: 'nested-before' }, linked, { type: 'tool_result', toolUseId: 'deep', content: [
        { type: 'image', source: { type: 'base64', mediaType: 'image/png', data: sampleImageData } },
        { type: 'text', text: 'nested-after' },
      ] },
    ] }] };
    expect(hasImageInToolResults([result])).toBe(true);
    const split = formatToolResultsForSplitTurn([result]);
    expect(split.hasImages).toBe(true);
    expect(split.userContent.filter(block => block.type === 'image')).toEqual([linked, { type: 'image', source: {
      type: 'base64', media_type: 'image/png', data: sampleImageData,
    } }]);
    const xml = split.beforeImageXml + split.userContent.filter(block => block.type === 'text').map(block => block.text).join('') + split.afterImageXml;
    expect(xml).toContain('nested-before');
    expect(xml).toContain('nested-after');
    expect(xml).not.toContain(sampleImageData);
    expect(formatToolResults([result])).not.toContain(sampleImageData);
  });

  it('malformed nested base64 becomes a bounded unavailable slot instead of a visual or byte-bearing XML part', () => {
    const result: ToolResult = { toolUseId: 'outer', content: [{ type: 'tool_result', toolUseId: 'nested', content: [
      { type: 'image', source: { type: 'base64', mediaType: 'image/png', data: '%%%%' } },
    ] }] };
    expect(hasImageInToolResults([result])).toBe(false);
    const split = formatToolResultsForSplitTurn([result]);
    expect(split.hasImages).toBe(false);
    expect(split.userContent).toEqual([]);
    const xml = formatToolResults([result]);
    expect(xml).not.toContain('%%%%');
    expect(xml.length).toBeLessThan(500);
  });

  it('degrades to complete XML with no images', () => {
    const split = formatToolResultsForSplitTurn([createTextResult('tool_1', 'Hello world')]);
    expect(split.hasImages).toBe(false);
    expect(split.userContent).toEqual([]);
    expect(split.afterImageXml).toBe('');
    expect(split.beforeImageXml).toContain('<function_results>');
    expect(split.beforeImageXml).toContain('</function_results>');
  });

  it('splits a single image result around the image', () => {
    const split = formatToolResultsForSplitTurn([createImageResult('tool_1', 'Screenshot taken')]);
    expect(split.hasImages).toBe(true);
    expect(split.userContent).toHaveLength(1);
    // Before: opening tags + text, held open at the image point.
    expect(split.beforeImageXml).toContain('<function_results>');
    expect(split.beforeImageXml).toContain('<tool_name>name_tool_1</tool_name>');
    expect(split.beforeImageXml).toContain('Screenshot taken');
    expect(split.beforeImageXml).not.toContain('</result>');
    expect(split.beforeImageXml).not.toContain('</stdout>');
    // After: the closers.
    expect(split.afterImageXml).toContain('</stdout>');
    expect(split.afterImageXml).toContain('</result>');
    expect(split.afterImageXml).toContain('</function_results>');
    expect(split.userContent).toEqual([{ type: 'image', source: {
      type: 'base64', media_type: 'image/png', data: sampleImageData,
    } }]);
  });

  it('splits at the first image when the FIRST of two results carries it', () => {
    const split = formatToolResultsForSplitTurn([
      createImageResult('tool_1', 'First result with image'),
      createTextResult('tool_2', 'Second result text only'),
    ]);
    expect(split.hasImages).toBe(true);
    expect(split.userContent).toHaveLength(1);
    expect(split.beforeImageXml).toContain('<tool_name>name_tool_1</tool_name>');
    expect(split.beforeImageXml).toContain('First result with image');
    // The full second result lands after the image.
    expect(split.afterImageXml).toContain('</result>');
    expect(split.afterImageXml).toContain('<tool_name>name_tool_2</tool_name>');
    expect(split.afterImageXml).toContain('Second result text only');
    expect(split.afterImageXml).toContain('</function_results>');
  });

  it('splits at the image when the SECOND of two results carries it', () => {
    const split = formatToolResultsForSplitTurn([
      createTextResult('tool_1', 'First result text only'),
      createImageResult('tool_2', 'Second result with image'),
    ]);
    expect(split.hasImages).toBe(true);
    expect(split.userContent).toHaveLength(1);
    // The complete first result + second result's text ride before the image.
    expect(split.beforeImageXml).toContain('<tool_name>name_tool_1</tool_name>');
    expect(split.beforeImageXml).toContain('First result text only');
    expect(split.beforeImageXml).toContain('</result>');
    expect(split.beforeImageXml).toContain('<tool_name>name_tool_2</tool_name>');
    expect(split.beforeImageXml).toContain('Second result with image');
    expect(split.afterImageXml).toContain('</result>');
    expect(split.afterImageXml).toContain('</function_results>');
  });

  it('carries all images of a multi-image result', () => {
    const split = formatToolResultsForSplitTurn([
      createMultiImageResult('tool_1', 'Multiple screenshots', 3),
    ]);
    expect(split.hasImages).toBe(true);
    expect(split.userContent).toHaveLength(3);
    expect(split.userContent).toEqual(Array.from({ length: 3 }, () => ({
      type: 'image', source: { type: 'base64', media_type: 'image/png', data: sampleImageData },
    })));
  });

  it('uses the error tag for error results with images', () => {
    const errorResult: ToolResult = {
      toolUseId: 'tool_1',
      content: [
        { type: 'text', text: 'Error occurred' },
        { type: 'image', source: { type: 'base64', data: sampleImageData, mediaType: 'image/png' } },
      ],
      isError: true,
    };
    const split = formatToolResultsForSplitTurn([errorResult]);
    expect(split.hasImages).toBe(true);
    expect(split.beforeImageXml).toContain('<error');
    expect(split.afterImageXml).toContain('</error>');
  });

  it('reassembles into balanced XML around the image slot', () => {
    const split = formatToolResultsForSplitTurn([createImageResult('tool_1', 'Test content')]);
    const fullXml = split.beforeImageXml + '[IMAGE]' + split.afterImageXml;
    expect(fullXml.match(/<function_results>/g)).toHaveLength(1);
    expect(fullXml.match(/<\/function_results>/g)).toHaveLength(1);
    expect((fullXml.match(/<result>/g) || []).length).toBe(
      (fullXml.match(/<\/result>/g) || []).length,
    );
  });
});
