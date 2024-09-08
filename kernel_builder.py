import argparse
import subprocess
import os
import shutil
import re
import time
import logging
from datetime import datetime
import zipfile
from enum import Enum
import configparser
from collections import OrderedDict

debug_popen_impl = False
debug_popen_impl_writefile = False

# Set default encoding to UTF-8
UTF8Codec = "utf-8"

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def popen_impl(command: list[str]):
    if debug_popen_impl:
        logging.debug("Executing command: %s", " ".join(command))
    s = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = s.communicate()

    def write_logs(out, err):
        out = out.decode(UTF8Codec)
        err = err.decode(UTF8Codec)
        stdout_log = str(s.pid) + "_stdout.log"
        stderr_log = str(s.pid) + "_stderr.log"
        with open(stdout_log, "w") as f:
            f.write(out)
        with open(stderr_log, "w") as f:
            f.write(err)
        logging.info(f"Output log files: {stdout_log}, {stderr_log}")

    if s.returncode != 0:
        if debug_popen_impl:
            print("failed")
        write_logs(out, err)
        raise RuntimeError(f"Command failed: {command}. Exitcode: {s.returncode}")
    if debug_popen_impl:
        logging.debug(f"result: {s.returncode == 0}")
        if debug_popen_impl_writefile:
            write_logs(out, err)


def match_and_get(regex: str, pattern: str):
    matched = re.search(regex, pattern)
    if not matched:
        raise AssertionError(
            "Failed to match: for pattern: %s regex: %s" % (pattern, regex)
        )
    return matched.group(1)


class CompilerTest:
    def __init__(self, exe_path: str, versionRegex: str) -> None:
        self.versionRegex = versionRegex
        self.versionArgList = [exe_path, "--version"]

    def test_executable(self):
        try:
            popen_impl(self.versionArgList)
            return True
        except Exception as e:
            logging.error(f"Failed to execute: {e}")
            return False

    def get_version(self):
        s = subprocess.Popen(
            self.versionArgList, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        tcversion, _ = s.communicate()
        tcversion = tcversion.decode(UTF8Codec)
        return match_and_get(self.versionRegex, tcversion)

    def get_path(self) -> str:
        return self.versionArgList[0]

    # TODO
    def is_clang(self) -> bool:
        return self.get_path().endswith("clang")


class KernelArch(Enum):
    x86_64 = "x86_64"
    arm64 = "arm64"
    arm = "arm"
    x86 = "x86"

    @classmethod
    def from_str(cls, arch_str: str) -> "KernelArch":
        return cls[arch_str.lower()]

    def to_str(self):
        return self.name


class KernelType(Enum):
    Image = "Image"
    zImage = "zImage"
    Image_dtb = "Image-dtb"
    Image_gz_dtb = "Image.gz-dtb"

    @classmethod
    def from_str(cls, kernel_type_str: str) -> "KernelType":
        return cls[kernel_type_str]

    def to_str(self):
        return self.name


class KernelConfig:
    def _parse_info(self):
        self.name = self.config.get("info", "Name")
        self.repo_url = self.config.get("info", "RepoUrl")
        self.repo_branch = self.config.get("info", "RepoBranch")
        self.kernel_arch = KernelArch.from_str(self.config.get("info", "KernelArch"))
        self.kernel_type = KernelType.from_str(self.config.get("info", "KernelType"))

    def _parse_defconfig(self):
        self.namingscheme = self.config.get("defconfig", "NamingScheme")
        self.devices = self.config.get("defconfig", "Devices").split(",")

    def _parse_config_fragments(self):
        self.config_fragments = []
        for section in self.config.sections():
            if section.startswith("config_fragment"):

                def split_get(key: str):
                    val = self.config.get(section, key, fallback=None)
                    return val.split(",") if val else None

                fragment = {
                    "NamingScheme": self.config.get(section, "NamingScheme"),
                    "TargetDevice": split_get("TargetDevice"),
                    "DependsOn": split_get("DependsOn"),
                    "Description": self.config.get(section, "Description"),
                }
                self.config_fragments.append(fragment)

    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        assert os.path.isfile(
            self.config_file
        ), f"Kernelconfig file not found: {self.config_file}"
        self.config.read(self.config_file)
        self._parse_info()
        self._parse_defconfig()
        self._parse_config_fragments()
        logging.debug(f"Config parsed successfully: {self.name}")

    def clone(self):
        popen_impl(
            [
                "git",
                "clone",
                self.repo_url,
                "--depth=1",
                "-b",
                self.repo_branch,
                self.name.replace(" ", "_"),
            ]
        )

    def dest_dir(self) -> str:
        return self.name.replace(" ", "_")


def get_gcc_prefixes(kernel_arch: KernelArch) -> list[str]:
    gcc_prefixes = {
        KernelArch.x86_64: ["x86_64-linux-gnu-", "x86_64-linux-android-"],
        KernelArch.arm64: ["aarch64-linux-gnu-", "aarch64-linux-android-"],
        KernelArch.arm: ["arm-linux-gnueabihf-", "arm-linux-androideabi-"],
        KernelArch.x86: ["i686-linux-gnu-", "i686-linux-android-"],
    }
    return gcc_prefixes.get(kernel_arch)


def check_file(filename: str, existFn):
    # Log that you're checking the file existence
    logging.info(f"Checking if file exists: {filename}")
    exists = existFn(filename)

    # Log the result of the file check
    if not exists:
        logging.warning(f"File not found: {filename}")
    else:
        logging.info(f"File found: {filename}")

    return exists


def zip_files(zipfilename: str, files: list[str]):
    logging.info(f"Zipping {len(files)} files to {zipfilename}...")
    with zipfile.ZipFile(zipfilename, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            zf.write(f)
    logging.info("Done")


def choose_from_list(choices: list[str]) -> str:
    while True:
        for i, choice in enumerate(choices, start=1):
            print(f"{i}. {choice}")
        choice = input("Choose a device (1-" + str(len(choices)) + "): ")
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(choices):
                return choices[choice_index]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid choice. Please enter a number.")


def main():
    # Parse the main ini file.
    iniFile = os.path.join("configs", "kernelbuilder.ini")
    assert os.path.isfile(iniFile), "kernelbuilder.ini not found"
    mainConfig = configparser.ConfigParser()
    mainConfig.read(iniFile)

    # Load popen_impl debug settings
    global debug_popen_impl, debug_popen_impl_writefile
    debug_popen_impl = mainConfig.getboolean("popen_impl", "Debug", fallback=False)
    debug_popen_impl_writefile = mainConfig.getboolean(
        "popen_impl", "WriteLogFiles", fallback=False
    )

    # Load directories config
    toolchainDirectory = mainConfig.get("directory", "Toolchain")
    AnyKernelDirectory = mainConfig.get("directory", "AnyKernel")
    OutDirectory = mainConfig.get("directory", "Out")

    # Parse the kernel specific ini files.
    kernelConfigDir = os.path.join("configs", "kernels", "")
    kernelConfigFiles = [f for f in os.listdir(kernelConfigDir) if f.endswith(".ini")]
    kernelConfigs = []
    for kernelConfigFile in kernelConfigFiles:
        kernelConfig = KernelConfig(os.path.join(kernelConfigDir, kernelConfigFile))
        kernelConfigs.append(kernelConfig)
    logging.info(f"Parsed {len(kernelConfigs)} kernel configurations")

    # Ask user what device to build for.
    device_choices = []
    for config in kernelConfigs:
        device_choices += config.devices
    device_choice = choose_from_list(device_choices)
    logging.info(f"Building for device: {device_choice}")

    # TODO: Maybe support multiple configs per device.
    selectedKernelConfig = [
        config for config in kernelConfigs if device_choice in config.devices
    ][0]
    logging.info(f"Selected kernel: {selectedKernelConfig.name}")

    # Ask user if they want to clone the Kernel repo.
    kernelDirectory = input(
"""Do you want to clone the Kernel repo?
If no, provide a directory with the kernel clone, else just hit enter: """
    )
    if kernelDirectory.strip() == "":
        isCloning = not os.path.isdir(selectedKernelConfig.dest_dir())
        if isCloning:
            logging.info("Cloning Kernel repo...")
            selectedKernelConfig.clone()
        os.chdir(selectedKernelConfig.dest_dir())
        if isCloning:
            logging.info("Done")
    else:
        logging.info(f"Using provided kernel directory: {kernelDirectory}")
        os.chdir(kernelDirectory)

    defconfig_list = [
        selectedKernelConfig.namingscheme.replace("{device}", device_choice)
    ]

    # Check if user wants additional kernel config fragments.
    def ask_and_append(frags):
        if frags["DependsOn"] is not None:
            print(f"[Depends on: {', '.join(frags['DependsOn'])}]", end=" ")
        x = input(f"Apply config fragment '{frags['Description']}'? (Y/n): ")
        if x.lower() == "y" or x.lower() == "n" or x == "":
            if x.lower() == "y" or x.lower() == "":
                print("Answered Y.")
                defconfig_list.append(
                    frags["NamingScheme"].replace("{device}", device_choice)
                )
            else:
                print(f"Answered N.")
        else:
            print("Invalid input.")

    def is_in_target(frags) -> bool:
        return frags["TargetDevice"] is None or device_choice in frags["TargetDevice"]

    # First, apply fragments without requirement.
    for frags in selectedKernelConfig.config_fragments:
        if is_in_target(frags) and frags["DependsOn"] is None:
            ask_and_append(frags)

    # Second, apply fragments with requirements. Although this would not work for more than
    # Three relation trees...
    for frags in selectedKernelConfig.config_fragments:
        if (
            is_in_target(frags)
            and frags["DependsOn"] is not None
            and all(depends in defconfig_list for depends in frags["DependsOn"])
        ):
            ask_and_append(frags)

    # Check if clang and gcc are available.
    compilers = []
    ClangCompilerTest = CompilerTest(
        os.path.join(toolchainDirectory, "bin", "clang"),
        r"(.*?clang version \d+(\.\d+)*).*",
    )
    compilers.append(ClangCompilerTest)
    compilers += [
        CompilerTest(path, r"(.*?gcc \(.*\) \d+(\.\d+)*)")
        for path in [
            os.path.join(toolchainDirectory, "bin", prefix + "gcc")
            for prefix in get_gcc_prefixes(selectedKernelConfig.kernel_arch)
        ]
    ]

    # Check if any compiler is available.
    selectedCompiler = None
    targetTriple = ""
    for compiler in compilers:
        if not compiler.test_executable():
            logging.warning(
                f"Unable to find {os.path.basename(compiler.get_path())} compiler"
            )
        else:
            selectedCompiler = compiler
            break
    if selectedCompiler is None:
        logging.fatal("No available compiler found")
        return
    else:
        logging.info(f"Selected compiler: {selectedCompiler.get_version()}")
        # Choose a target triple.
        if selectedCompiler.is_clang():
            match selectedKernelConfig.kernel_arch:
                case KernelArch.x86_64:
                    targetTriple = "x86_64-linux-gnu-"
                case KernelArch.x86:
                    targetTriple = "mipsel-linux-gnu-"
                case KernelArch.arm64:
                    targetTriple = "aarch64-linux-gnu-"
                case KernelArch.arm:
                    targetTriple = "arm-linux-gnueabi-"
            logging.debug(f"Target triple: {targetTriple}")
        else:
            logging.debug(
                f"Target triple: {os.path.basename(selectedCompiler.get_path())[:-4]}"
            )

    # Init submodules, if any.
    if os.path.isfile(".gitmodules"):
        try:
            popen_impl(["git", "submodule", "update", "--init"])
            logging.info("Submodules initialized successfully")
        except RuntimeError:
            pass

    arch = selectedKernelConfig.kernel_arch.to_str()
    type = selectedKernelConfig.kernel_type.to_str()

    # Add toolchain in PATH environment variable
    tcPath = os.path.join(os.getcwd(), toolchainDirectory, "bin")
    if tcPath not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = tcPath + ":" + os.environ["PATH"]

    if os.path.exists(OutDirectory) and not False:
        logging.info("Make clean...")
        shutil.rmtree(OutDirectory)

    make_defconfig = []
    make_common = [
        "make",
        "ARCH=" + arch,
        "O=" + OutDirectory,
        "CROSS_COMPILE=" + targetTriple,
        "LLVM=1",
        f"-j{os.cpu_count()}",
    ]
    make_defconfig += make_common
    make_defconfig += defconfig_list

    t = datetime.now()
    logging.info("Make defconfig...")
    popen_impl(make_defconfig)
    logging.info("Make kernel...")
    popen_impl(make_common)
    logging.info("Done")
    t = datetime.now() - t

    with open(os.path.join(OutDirectory, "include", "generated", "utsrelease.h")) as f:
        kver = match_and_get(r'"([^"]+)"', f.read())

    shutil.copyfile(
        "out/arch/" + arch + "/boot/" + type,
        os.path.join(AnyKernelDirectory, type),
    )
    zipname = (selectedKernelConfig.dest_dir() + "_{}_{}.zip").format(
        device_choice, datetime.today().strftime("%Y-%m-%d")
    )
    os.chdir(AnyKernelDirectory)
    zip_files(
        zipname,
        [
            selectedKernelConfig.kernel_type.to_str(),
            "META-INF/com/google/android/update-binary",
            "META-INF/com/google/android/updater-script",
            "tools/ak3-core.sh",
            "tools/busybox",
            "tools/magiskboot",
            "anykernel.sh",
            "version",
        ],
    )
    newZipName = os.path.join(os.getcwd(), "..", zipname)
    try:
        os.remove(newZipName)
    except:
        pass
    shutil.move(zipname, newZipName)
    os.chdir("..")
    logging.info("Kernel zip created:", newZipName)
    logging.info("Kernel version:", kver)
    logging.info("Elapsed time:", str(t.total_seconds()) + " seconds")


if __name__ == "__main__":
    main()
