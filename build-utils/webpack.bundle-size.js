const BundleAnalyzerPlugin = require("webpack-bundle-analyzer")
    .BundleAnalyzerPlugin;

module.exports.config = ({ file }) => {
    return {
        plugins: [new BundleAnalyzerPlugin()]
    };
};
