const WebpackMerge = require("webpack-merge");

const targetConfig = env => {
	const path = `./build-utils/webpack.${env.mode}.js`;
	return require(path).config(env);
};

module.exports = env => {
	return WebpackMerge(
		{
			entry: "./src/main.ts",
			resolve: {
				extensions: [".js", ".jsx", ".json", ".ts", ".tsx"]
			},
			module: {
				rules: [
					{
						test: /\.(ts|tsx)$/,
						use: "ts-loader"
					}
				]
			}
		},
		targetConfig(env)
	);
};
