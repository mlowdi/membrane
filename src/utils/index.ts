/**
 * Utility exports
 */

export {
  parseToolCalls,
  formatToolResults,
  formatToolResult,
  formatToolDefinitions,
  getToolInstructions,
  hasUnclosedToolBlock,
  endsWithPartialToolBlock,
  unescapeXml,
  type ToolDefinitionForPrompt,
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
  isValidImageBase64,
  projectResponsesItem,
  projectNativeImageContent,
  estimateImagePolicyContentTokens,
} from './image-policy.js';
export type { LiveImagePolicy, ImageMessageFilter } from './image-policy.js';
