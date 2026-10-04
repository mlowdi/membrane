/**
 * Utility exports
 */

export {
  parseToolCalls,
  formatToolResults,
  formatToolResult,
  formatToolDefinitions,
  toolDefinitionForPrompt,
  getToolInstructions,
  hasUnclosedToolBlock,
  endsWithPartialToolBlock,
  unescapeXml,
  type ToolDefinitionForPrompt,
  type ToolParseOptions,
} from './tool-parser.js';

export { calculateCost } from './cost.js';
export type { CostableUsage } from './cost.js';
export {
  DEFAULT_MAX_LIVE_IMAGE_BYTES,
  IMAGE_TOKEN_ESTIMATE,
  IMAGE_DROPPED_TEXT,
  IMAGE_UNAVAILABLE_TEXT,
  imagePayloadBytes,
  imageDepthStart,
  filterImageMessages,
  createImageMessageFilter,
  projectResponsesContent,
  normalizeImageContent,
  isVisualImageContent,
  hasVisualImageContent,
  isImageReference,
  isGeneratedImageMetadata,
  asImageContent,
  isValidImageBase64,
  projectResponsesItem,
  projectResponsesGeneratedImage,
  projectNativeImageContent,
  estimateImagePolicyContentTokens,
} from './image-policy.js';
export type { LiveImagePolicy, ImageMessageFilter, GeneratedImageMetadata } from './image-policy.js';
