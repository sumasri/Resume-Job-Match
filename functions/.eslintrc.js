module.exports = {
  root: true,
  env: {
    es6: true,
    node: true,
  },
  parserOptions: {
    ecmaVersion: 2018,
  },
  extends: ["eslint:recommended"],
  rules: {
    "no-unused-vars": ["error", { argsIgnorePattern: "next" }],
    "no-restricted-globals": ["error", "name", "length"],
    "prefer-arrow-callback": "error",
  },
};
