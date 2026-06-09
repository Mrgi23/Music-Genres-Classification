import utils_cpp
import argparse
import tomli
from pathlib import Path

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Application")

  parser.add_argument("-p", "--predict", type=str, default="None", help="sound file path to predict genre. If not provided, model is evaluated on the test dataset")
  parser.add_argument("-f", "--force", action="store_true", help="force model training")
  parser.add_argument("-s", "--save", action="store_true", help="save new model (only if force training)")

  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  with Path("./config/config.toml").open("rb") as config_file:
    config = tomli.load(config_file)
    assets = config["ASSETS"]
    preprocessor_cfg = config["PREPROCESSOR"]
    model_cfg = config["MODEL"]
    scheduler_cfg = config["SCHEDULER"]
    trainer_cfg = config["TRAINER"]

  train_dataset, val_dataset, test_dataset = utils_cpp.init(preprocessor_cfg, assets["DATASET"], assets["SPLITS"])

  if args.force:
    model = utils_cpp.train(trainer_cfg, scheduler_cfg, train_dataset, val_dataset)

    if args.save:
      model.save(assets["MODEL_CPP"])
  else:
    model = utils_cpp.load_model(model_cfg, Path("./") / assets["MODEL_CPP"])

  if Path(args.predict).exists() and Path(args.predict).rglob("*.wav"):
    preprocessor = train_dataset.dataset().preprocessor()
    classes = train_dataset.dataset().classes()
    utils_cpp.predict(model, preprocessor, args.predict, classes)
  else:
    utils_cpp.evaluate(trainer_cfg, model, test_dataset)
