import sys
import os

from json import load

from optimizer.rom.rom_main import rom_main
from optimizer.logger import get_logger, get_logs_directory


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = get_logger(__name__)


def main(config_path, direct_json=False):
    logger.info(f'Logs directory: {get_logs_directory()}')
    if direct_json:
        json_config = load(open(config_path))
    else:
        with open(config_path) as f:
            json_config = load(f)
    run = {"ROM": rom_main}
    run[json_config["module"]](config_path)


if __name__ == "__main__":
    main("E:/llm-rom/src/rom_llm/config.json", direct_json=False)
    # main("E:\\llm-rom\\rom_llm\\train_config.json", direct_json=False)
    # freeze_support()
    # if len(sys.argv) != 2:
    #     print("Использование: program.exe <путь_к_конфигурационному_файлу>")
    #     input("Нажмите Enter для выхода...")
    #     sys.exit(1)

    # config_path = sys.argv[1]

    # try:
    #     main(config_path)
    # except KeyboardInterrupt:
    #     print("\nПрервано пользователем")
    # except Exception as e:
    #     logger.exception("Произошла ошибка:")
    #     print(f"\nКритическая ошибка: {str(e)}")
    # finally:
    #     input("\nНажмите Enter для закрытия программы...")
        