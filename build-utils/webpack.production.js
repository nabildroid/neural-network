const glob = require("glob");
const WebpackMerge = require("webpack-merge");

const PurgecssPlugin = require("purgecss-webpack-plugin");
const TerserJSPlugin = require("terser-webpack-plugin");
const OptimizeCSSAssetsPlugin = require("optimize-css-assets-webpack-plugin");

const prodAndDev = require("./prod-dev");

module.exports.config = () =>
  WebpackMerge(
    {
      mode: "production",
      optimization: {
        minimizer: [new TerserJSPlugin({}), new OptimizeCSSAssetsPlugin({})],
        splitChunks: {
          cacheGroups: {
            styles: {
              name: "styles",
              test: /\.css$/,
              chunks: "all",
              enforce: true,
            },
          },
        },
      },
      plugins: [
        new PurgecssPlugin({
          paths: glob.sync(`src/**/*`, { nodir: true }),
        }),
      ],
      // devtool: "eval"
    },
    prodAndDev
  );
