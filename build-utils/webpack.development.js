const WebpackMerge = require("webpack-merge");
const prodAndDev = require("./prod-dev");


module.exports.config = () =>
	new WebpackMerge(
		{
			mode: "development",
			devServer: {
			  host: 'localhost',
			  historyApiFallback: true,
			  open: true
			}
		},
		prodAndDev
	);
