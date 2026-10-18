import { defineConfig, globalIgnores } from "eslint/config";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export default defineConfig([
  ...tseslint.configs.recommended,
  reactHooks.configs.flat.recommended,
  globalIgnores([".next/**", ".npm-cache/**", "artifacts/**", "next-env.d.ts"]),
]);
