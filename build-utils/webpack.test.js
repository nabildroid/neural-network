module.exports.config = ({ file }) => {
    return {
        mode: "production",
        entry: `./tests-dev/test.${file}.ts`,
        stats: "errors-only"
    };
};
