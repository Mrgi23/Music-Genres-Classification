#include "utils.h"

/**
 * @brief Command line arguments storage.
 */
struct Args
{
    bool help = false;
    fs::path workingDir = fs::absolute("./");
    fs::path predict = "";
    bool force = false;
    bool save = false;
};

/**
 * @brief Parse command line arguments into an Args structure.
 *
 * Supported options:
 *
 *   - `-h`, `--help`:
 *       Displays help information. If provided, application prints
 *       help output and stops.
 *
 *   - `-wd`, `--working-dir <path>`:
 *       Specifies the working directory. The provided path is resolved
 *       to an absolute path. If omitted or the next token is another flag,
 *       an exception is thrown.
 *
 *   - `-p`, `--predict <file>`:
 *       Specifies an audio file to run prediction on. The provided path
 *       is resolved to an absolute path. If omitted or the next token is
 *       another flag, an exception is thrown.
 *
 *   - `-f`, `--force`:
 *       Enables overwrite mode (e.g., allows replacing existing files).
 *
 *   - `-s`, `--save`:
 *       Enables saving of results (e.g., trained model checkpoints).
 *
 * @param[in] argc Number of command line arguments.
 * @param[in] argv Array of command line argument strings.
 *
 * @return Args Populated Args structure containing parsed options.
 *
 * @throws std::invalid_argument If a required value (working directory
 *                               or prediction file) is missing or invalid.
 */
Args ParseArgs(int argc, char * argv[])
{
    // Parse command line arguments.
    Args args;
    for (uint i = 0; i < argc; i++)
    {
        string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            args.help = true;
            return args;
        }
        else if (arg == "-wd" || arg == "--working-dir")
        {
            if (i + 1 >= argc || string(argv[i + 1]).starts_with("-"))
                throw invalid_argument("ParseArgs: Working directory not specified.");
            args.workingDir = fs::absolute(fs::path(argv[++i]));
        }
        else if (arg == "-p" || arg == "--predict")
        {
            if (i + 1 >= argc || string(argv[i + 1]).starts_with("-"))
                throw invalid_argument("ParseArgs: Prediction file not specified.");
            args.predict = fs::absolute(fs::path(argv[++i]));
        }
        else if (arg == "-f" || arg == "--force")
        {
            args.force = true;
        }
        else if (arg == "-s" || arg == "--save")
        {
            args.save = true;
        }
    }
    return args;
}

int main(int argc, char * argv[])
{
    try
    {
        // Parse command line arguments.
        Args args = ParseArgs(argc, argv);

        // Print help
        if (args.help)
        {
            PrintHelp();
            return 0;
        }

        // Set working dir.
        if (fs::exists(args.workingDir))
            global::root = args.workingDir;

        // Load configuration.
        fs::path configPath = GetFullPath("config/config.toml");
        toml::table config = toml::parse_file(configPath.string());
        auto& assets = *config["ASSETS"].as_table();
        auto& preprocessorCfg = *config["PREPROCESSOR"].as_table();
        auto& modelCfg = *config["MODEL"].as_table();
        auto& schedulerCfg = *config["SCHEDULER"].as_table();
        auto& trainerCfg = *config["TRAINER"].as_table();

        // Initialization
        Init(preprocessorCfg, GetFullPath(GetParam<string>(assets, "DATASET")), GetFullPath(GetParam<string>(assets, "SPLITS")));

        if (args.force)
        {
            // Force train new model.
            LoadScheduler(schedulerCfg);
            Train(trainerCfg);

            // Save new model.
            if (args.save)
                global::model->Save(GetFullPath(GetParam<string>(assets, "MODEL_CPP")));
        }
        else
            // Load trained model.
            LoadModel(modelCfg, GetFullPath(GetParam<string>(assets, "MODEL_CPP")));

        if (fs::exists(args.predict) && args.predict.extension() == ".wav")
            // Predict genre for new entry.
            Predict(args.predict);
        else
            // Evaluate model on test dataset.
            Evaluate(trainerCfg);

        cleanUp();
        return 0;
    }
    catch (const exception & e)
    {
        cerr << e.what() << endl;
        cleanUp();
        return 1;
    }
}