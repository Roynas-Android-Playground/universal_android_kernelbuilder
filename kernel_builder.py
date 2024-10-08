import configparser
import subprocess
import os
import shutil
import sys
import re
import logging
import time
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(filename)s:%(lineno)-3d - %(levelname)-8s: %(message)s",
)


class DiagnoseLevel(Enum):
    Error = "err"
    Warning = "warning"
    Note = "note"

    @classmethod
    def from_str(cls, s: str):
        return cls[s.capitalize()]


diagnose_re = re.compile(
    r"([\/\w\.-]+):(\d+):(\d+):\s+(warning|error|note):\s+([\w ;‘’?|'\[\]\"()\-.:]+)\s(\[(-[\w-]+)(=)?\])?"
)
undefined_symbol_gnuld_re = re.compile(r"undefined reference to `(\w+)'")
undefined_symbol_lld_re = re.compile(r"ld\.lld: error: undefined symbol: ([\w ()*]+)")


def parse_diagnose_msg(line: str) -> tuple[int, int, DiagnoseLevel, str, str]:
    matches = diagnose_re.match(line)
    if matches:
        try:
            file_path = matches.group(1)
            line_number = int(matches.group(2))
            column_number = int(matches.group(3))
            message_type = DiagnoseLevel.from_str(matches.group(4))
            message = matches.group(5)
            if matches.group(7):
                warning_code = matches.group(7)
            else:
                warning_code = "Unknown"
                if '-W' in line:
                    logging.warning(f'Couldn\'t match warning message, text is: {line}')

            return (
                file_path,
                line_number,
                column_number,
                message_type,
                message,
                warning_code,
            )
        except (ValueError, KeyError) as e:
            logging.error("Error parsing diagnose message: %s", str(e))

    return None, None, None, None, None, None


def parse_linkage_fail(line: str):
    # Use re.search here...
    matches = undefined_symbol_gnuld_re.search(line)
    if matches:
        return matches.group(1)
    matches = undefined_symbol_lld_re.match(line)
    if matches:
        return matches.group(1)
    return None


class PopenImpl:
    class DebugMode(Enum):
        Off = 0
        Debug = 1
        Debug_OutputToFile = 2

        def isDebug(self) -> bool:
            return self != self.Off

    debugmode = DebugMode.Off

    def run(self, command: list[str], **kwargs):
        """
        Execute a command using subprocess.Popen and handle its output and errors.

        Parameters:
        command (list[str]): The command to be executed.
        **kwargs: Additional keyword arguments for subprocess.Popen.
        timed_output (bool) in **kwargs: If True, the command output is shown.

        Raises:
        RuntimeError: If the command execution fails.
        OSError: subprocess.Popen would throw that additionally.

        Returns:
        Tuple[str, str]: The command's standard output and standard error.
        """
        if self.debugmode.isDebug():
            logging.debug("Executing command: %s", " ".join(command))

        use_timed_output = kwargs.get("timed_output", None)
        if use_timed_output:
            kwargs.pop("timed_output")
        EXTENDED_DIAGNOSIS = kwargs.get("ext_diag", None)
        if EXTENDED_DIAGNOSIS:
            kwargs.pop("ext_diag")

        s = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
        )

        if use_timed_output:
            # Initialize variables to store the output
            out = []
            err = []

            start = datetime.now()
            # Read stdout in real-time with delay and carriage return
            while True:
                # Read a line from stdout
                stdout_line = s.stdout.readline()
                ts = (datetime.now() - start).total_seconds()
                if stdout_line and round(ts) % 5 == 0:
                    for line in stdout_line.splitlines():
                        print(
                            f"\r\033[K[{ts}s] {line.strip()}", end='', flush=True
                        )  # Print the stdout line with carriage return
                out.append(stdout_line)  # Collect the stdout output

                # If the process is finished and stdout is empty, break the loop
                if s.poll() is not None and stdout_line == "":
                    break

            print()
            remaining_out, remaining_err = s.communicate()

            if remaining_out:
                out.append(remaining_out)
            if remaining_err:
                err.append(remaining_err)
            out, err = "".join(out), "".join(err)
        else:
            out, err = s.communicate()

        if EXTENDED_DIAGNOSIS:
            # Handle the extended diagnostics mode
            def log_extended_diagnostics(level, message):
                logging.log(level, f"[LogXpert EXT] {message}")

            log_extended_diagnostics(logging.INFO, "Diagnosis reports")
            map = dict[str, int]()
            # Parse the diagnose messages from the output
            for line in err.splitlines():
                (
                    path,
                    line_number,
                    column_number,
                    message_type,
                    message,
                    warning_code,
                ) = parse_diagnose_msg(line)
                if path:
                    match message_type:
                        case DiagnoseLevel.Warning:
                            if path not in map:
                                map[path] = []
                            map[path].append(warning_code)
                        case DiagnoseLevel.Error:
                            log_extended_diagnostics(
                                logging.ERROR,
                                f"File {path}:{line_number}:{column_number} {message}",
                            )
                            log_extended_diagnostics(
                                logging.ERROR, f"Warning code: {warning_code}"
                            )
                else:
                    linkage_fail_msg = parse_linkage_fail(line)
                    if linkage_fail_msg:
                        log_extended_diagnostics(
                            logging.ERROR, f"UNDEFINED SYMBOL: {linkage_fail_msg}"
                        )
            for path, warnlist in map.items():
                uniquelist = list(set(warnlist))
                log_extended_diagnostics(
                    logging.WARNING,
                    f"File {path} has {len(warnlist)} warning(s). Kinds: {uniquelist}",
                )
            log_extended_diagnostics(logging.INFO, "End")

        def write_logs(out, err, failed=False):
            if self.debugmode == self.DebugMode.Debug_OutputToFile:
                logslist = []

                def write_one(buf: str, suffix: str):
                    if len(buf) == 0:
                        return

                    log = str(s.pid) + f"_{suffix}.log"
                    with open(log, "w") as f:
                        f.write(buf)
                    logslist.append(log)

                write_one(out, "stdout")
                write_one(err, "stderr")

                if len(logslist) > 0:
                    logging.info(f"Output log files: " + ", ".join(logslist))
            elif failed:
                logging.error("Failed, printing process stderr output to stderr...")
                print(err, file=sys.stderr)

        if s.returncode != 0:
            write_logs(out, err, failed=True)
            raise RuntimeError(f"Command failed: {command}. Exitcode: {s.returncode}")
        if self.debugmode.isDebug():
            logging.debug(f"result: {s.returncode == 0}")
            write_logs(out, err)

        return out, err

    def set_debugmode(self, mode: DebugMode) -> None:
        self.debugmode = mode

    def test_executable(self, path: Path, args: list[str] = []) -> bool:
        """
        Test if an executable file exists and can be executed with the provided arguments.

        Parameters:
        path (str): The path to the executable file.
        args (list[str]): The arguments to be passed to the executable.

        Returns:
        bool: True if the executable exists and can be executed with the provided arguments, False otherwise.
        """
        try:
            self.run([path.as_posix()] + args)
            return True
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning(f"Execution of {path} with args: {args} failed: {str(e)}")
            return False


_popen_impl_inst = PopenImpl()
popen_impl = _popen_impl_inst.run
popen_impl_set_debugmode = _popen_impl_inst.set_debugmode
popen_impl_test_executable = _popen_impl_inst.test_executable


def match_and_get_first(regex: str, pattern: str) -> str:
    """
    This function matches a given regex pattern with a provided string and returns the first group.

    Parameters:
    regex (str): The regular expression pattern to match.
    pattern (str): The string to match the regex pattern against.

    Returns:
    str: The first group matched by the regex pattern.

    Raises:
    AssertionError: If the regex pattern does not match the provided string.
    """
    matched = re.search(regex, pattern)
    if not matched:
        raise AssertionError(
            "Failed to match: for pattern: %s regex: %s" % (pattern, regex)
        )
    return matched.group(1)


class KernelArch(Enum):
    x86_64 = "x86_64"
    arm64 = "arm64"
    arm = "arm"
    x86 = "x86"

    @classmethod
    def from_str(cls, arch_str: str) -> "KernelArch":
        try:
            return cls[arch_str.lower()]
        except KeyError:
            raise ValueError(f"Unsupported architecture: {arch_str}")

    def to_str(self):
        return self.name


class ForEachArch(dict):
    def __init__(self, arm: str, arm64: str, x86: str, x86_64: str):
        super().__init__(
            {
                KernelArch.arm: arm,
                KernelArch.arm64: arm64,
                KernelArch.x86: x86,
                KernelArch.x86_64: x86_64,
            }
        )


class KernelType(Enum):
    Image = "Image"
    zImage = "zImage"
    Image_dtb = "Image-dtb"
    Image_gz_dtb = "Image.gz-dtb"

    @classmethod
    def from_str(cls, kernel_type_str: str) -> "KernelType":
        try:
            return cls[kernel_type_str]
        except KeyError:
            raise ValueError(f"Unsupported kernel type: {kernel_type_str}")

    def to_str(self):
        return self.name


class CompilerType(Enum):
    GCC = "gcc"
    Clang = "clang"
    GCCAndroid = "gcc-android"  # The old GCC 4.9 used in Android, now deprecated.

    def get_triples(self, arch: KernelArch) -> str:
        match self:
            case CompilerType.GCC | CompilerType.Clang:
                return CompilerType.common_triples[arch]
            case CompilerType.GCCAndroid:
                return CompilerType.android_triples[arch]
        raise ValueError(f"Unsupported compiler type: {self}")

    @staticmethod
    def determine_type(path: str) -> "CompilerType":
        exename = os.path.basename(path)
        if exename.endswith("gcc"):
            if "android" in exename:
                return CompilerType.GCCAndroid
            return CompilerType.GCC
        elif exename == "clang":
            return CompilerType.Clang
        else:
            raise ValueError(f"Unsupported compiler: {exename}")

    def extract_version(self, output: str) -> str:
        match self:
            case CompilerType.GCC | CompilerType.GCCAndroid:
                return match_and_get_first(self.gcc_version_regex, output)
            case CompilerType.Clang:
                return match_and_get_first(self.clang_version_regex, output)
        raise ValueError(f"Unsupported compiler type: {self}")

    def toolchain_version(self, tcPath: Path, arch: KernelArch) -> str:
        out, _ = popen_impl([str(tcPath / self.toolchain_exe(arch)), "--version"])
        return self.extract_version(out)

    def toolchain_exe(self, arch: KernelArch) -> str:
        match self:
            case CompilerType.GCC | CompilerType.GCCAndroid:
                return f"{self.get_triples(arch)}gcc"
            case CompilerType.Clang:
                return "clang"
        raise ValueError(f"Unsupported compiler type: {self}")


# Used by GCC/Clang
CompilerType.common_triples = ForEachArch(
    arm="arm-linux-gnueabi-",
    arm64="aarch64-linux-gnu-",
    x86="i686-linux-gnu-",
    x86_64="x86_64-linux-gnu-",
)

# Used by GCC-Android
CompilerType.android_triples = {
    key: value.replace("gnu", "android")
    for key, value in CompilerType.common_triples.items()
}

CompilerType.gcc_version_regex = r"(.*?gcc \(.*\) \d+(\.\d+)*)"
CompilerType.clang_version_regex = r"(.*?clang version \d+(\.\d+)*).*"


class ToolchainConfig(Enum):
    GCCOnly = "GCCOnly"
    GNUBinClang = "GNUBinClang"
    FullLLVM = "FullLLVM"
    FullLLVMWithIAS = "FullLLVMWithIAS"

    @classmethod
    def from_str(cls, toolchain_str: str) -> "ToolchainConfig":
        try:
            return cls[toolchain_str]
        except KeyError:
            raise ValueError(f"Unsupported toolchain configuration: {toolchain_str}")

    def get_config_list(self) -> list[str]:
        for item in self.makefile_command_variables:
            if item[0] == self:
                return [f"{key}={value}" for key, value in item[1].items()]


ToolchainConfig.makefile_command_variables = [
    (ToolchainConfig.GCCOnly, {}),  # No need to override any tools
    (
        ToolchainConfig.GNUBinClang,
        {
            "CC": "clang",
        },
    ),
    (
        ToolchainConfig.FullLLVM,
        {
            "CC": "clang",
            "LD": "ld.lld",
            "AR": "llvm-ar",
            "NM": "llvm-nm",
            "OBJCOPY": "llvm-objcopy",
            "OBJDUMP": "llvm-objdump",
        },
    ),
    (
        ToolchainConfig.FullLLVMWithIAS,
        {
            "LLVM": "1",
            # "LLVM_IAS": "1", TODO: Maybe?
        },
    ),
]


class KernelConfig:
    def _parse_info(self):
        def _get_info_element(key: str, **kwargs) -> str:
            return self.config.get("info", key, **kwargs)

        self.name = _get_info_element("Name")
        self.repo_url = _get_info_element("RepoUrl")
        self.repo_branch = _get_info_element("RepoBranch")
        simplename = _get_info_element("SimpleName", fallback=None)
        if simplename:
            self._simple_name = simplename
        else:
            # If SimpleName is not provided, use Name with spaces replaced by underscores
            self._simple_name = self.name.replace(" ", "_")
        self.envMap = _get_info_element("Env", fallback=None)
        if self.envMap:
            self.envMap = {
                k.strip(): v.strip()
                for k, v in (item.split("=") for item in self.envMap.split(","))
            }
        try:
            self.kernel_arch = KernelArch.from_str(_get_info_element("KernelArch"))
            self.kernel_type = KernelType.from_str(_get_info_element("KernelType"))
            self.toolchain_config = ToolchainConfig.from_str(
                _get_info_element("ToolchainConfig")
            )
        except ValueError as e:
            raise AssertionError(f"Invalid configuration value: {str(e)}")

    def _parse_defconfig(self):
        def _get_defconfig_element(key: str, **kwargs) -> str:
            return self.config.get("defconfig", key, **kwargs)

        self.namingscheme = _get_defconfig_element("NamingScheme")
        self.devices = _get_defconfig_element("Devices").split(",")

    def _parse_anykernel3(self):
        def _get_anykernel3_element(key: str, **kwargs) -> str:
            return self.config.get("anykernel3", key, **kwargs)

        directory = _get_anykernel3_element("Directory")
        if directory:
            self.anykernel3_directory = Path(directory)
            additionalFiles = _get_anykernel3_element("AdditionalFiles", fallback=None)
            self.anykernel3_addfiles = (
                [Path(file) for file in additionalFiles.split(",")]
                if additionalFiles
                else []
            )

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
                    "Default": self.config.getboolean(section, "Default"),
                }
                self.config_fragments.append(fragment)

    def __init__(self, config_file: Path) -> None:
        """
        Initialize a KernelConfig object with the given configuration file.

        Parameters:
        config_file (str): The path to the configuration file.

        Returns:
        None

        Raises:
        AssertionError: If the configuration file does not exist.
        configparser.Error: If the configuration file contains errors.
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        assert (
            self.config_file.is_file()
        ), f"Kernelconfig file not found: {self.config_file}"
        self.config.read(self.config_file)
        self._parse_info()
        self._parse_defconfig()
        self._parse_config_fragments()
        try:
            self._parse_anykernel3()
        except configparser.NoSectionError:
            self.anykernel3_directory = None
        logging.debug(f"Config parsed successfully: {self.name}")

    def clone(self, depth=None):
        argv = [
            "git",
            "clone",
            self.repo_url,
            "-b",
            self.repo_branch,
            self._simple_name,
        ]
        if depth:
            argv.append("--depth=" + str(depth))
        try:
            popen_impl(argv, timed_output=True)
        except RuntimeError as e:
            logging.error(f"Failed to clone repository: {e}")
            raise

    def simple_name(self) -> str:
        return self._simple_name


class UnImplementedError(Exception):
    pass


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
        try:
            choice = input("Choose a device (1-" + str(len(choices)) + "): ")
        except KeyboardInterrupt:
            print("\nAborting...")
            sys.exit(1)
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(choices):
                return choices[choice_index]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid choice. Please enter a number.")


def find_available_compilers(
    toolchain_directory: Path,
    kernel_arch: KernelArch,
    toolchain_config: ToolchainConfig,
):
    """
    Find available compilers based on the kernel architecture and toolchain configuration.

    Parameters:
    - toolchain_directory (Path): The path to the toolchain directory.
    - kernel_arch (KernelArch): The architecture of the kernel.
    - toolchain_config (ToolchainConfig): The toolchain configuration (GCCOnly, FullLLVM, etc.).

    Returns:
    - CompilerType: The selected compiler type.
    - config_list (list[str]): The list of configuration options for the selected compiler.
    - triples (str): The target triple for the selected compiler.
    """
    exe_dir = toolchain_directory / "bin"
    Ctype: CompilerType = None
    versionArgv = ["--version"]

    # Check the toolchain configuration
    match toolchain_config:
        case ToolchainConfig.GCCOnly:
            if popen_impl_test_executable(
                exe_dir / CompilerType.GCC.toolchain_exe(kernel_arch),
                versionArgv,
            ):
                Ctype = CompilerType.GCC
            elif popen_impl_test_executable(
                exe_dir / CompilerType.GCCAndroid.toolchain_exe(kernel_arch),
                versionArgv,
            ):
                Ctype = CompilerType.GCCAndroid
        case (
            ToolchainConfig.GNUBinClang
            | ToolchainConfig.FullLLVM
            | ToolchainConfig.FullLLVMWithIAS
        ):
            if popen_impl_test_executable(
                exe_dir / CompilerType.Clang.toolchain_exe(kernel_arch),
                versionArgv,
            ):
                Ctype = CompilerType.Clang

    if Ctype:
        return (
            Ctype,
            ToolchainConfig.get_config_list(toolchain_config),
            Ctype.get_triples(kernel_arch),
        )
    logging.error(f"Dump: Toolchain config: {toolchain_config}")
    raise UnImplementedError(f"Unsupported toolchain configuration")


from collections import defaultdict, deque


def build_dependency_graph(fragments):
    """
    Build the dependency graph for the kernel configuration fragments.

    Parameters:
    fragments (list[dict]): List of kernel configuration fragments.

    Returns:
    tuple[defaultdict, defaultdict]: A graph representing dependencies and a dictionary with in-degrees.
    """
    graph = defaultdict(list)  # Adjacency list for graph
    in_degree = defaultdict(
        int
    )  # Track in-degree of each fragment (number of dependencies)

    # Build the graph and in-degree map
    for fragment in fragments:
        # If a fragment has dependencies, add edges in the graph
        if fragment["DependsOn"] is not None:
            for dependency in fragment["DependsOn"]:
                # Add edge from dependency to fragment
                graph[dependency].append(fragment["NamingScheme"])
                # Increment in-degree for the fragment that depends on this
                in_degree[fragment["NamingScheme"]] += 1
        else:
            # Ensure all fragments are in the in-degree map
            if fragment["NamingScheme"] not in in_degree:
                in_degree[fragment["NamingScheme"]] = 0

    return graph, in_degree


def topological_sort(fragments):
    """
    Perform topological sorting on the fragments based on their dependencies.

    Parameters:
    fragments (list[dict]): List of kernel configuration fragments.

    Returns:
    list[dict]: A topologically sorted list of fragments.
    """
    graph, in_degree = build_dependency_graph(fragments)
    sorted_fragments = []

    # Initialize a queue with fragments that have no dependencies (in-degree 0)
    queue = deque([frag for frag in fragments if in_degree[frag["NamingScheme"]] == 0])

    while queue:
        current_fragment = (
            queue.popleft()
        )  # Get a fragment with no remaining dependencies
        sorted_fragments.append(current_fragment)

        # Reduce the in-degree of dependent fragments
        for dependent in graph[current_fragment["NamingScheme"]]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:  # If no more dependencies, add to queue
                queue.append(
                    next(
                        frag for frag in fragments if frag["NamingScheme"] == dependent
                    )
                )

    # Check if there's a cycle (unresolved dependencies)
    if len(sorted_fragments) != len(fragments):
        raise RuntimeError("Dependency cycle detected in fragments")

    return sorted_fragments


def ask_yesno(question, allow_empty=False):
    """
    Ask the user a yes/no question and return their response as a boolean.

    Parameters:
    question (str): The question to ask the user.

    Returns:
    bool: True if the user answered 'yes', False otherwise.
    """
    while True:
        try:
            response = input(question + " (yes/no): ").lower()
        except KeyboardInterrupt:
            print("\nAborting...")
            sys.exit(1)

        if allow_empty and response == "":
            return True
        elif response == "yes" or response == "y":
            return True
        elif response == "no" or response == "n":
            return False
        else:
            print("Invalid response. Please answer 'yes' or 'no'.")


def apply_fragments(selectedKernelConfig, device_choice):
    """
    Apply the kernel configuration fragments in the correct order of dependency resolution,
    while asking the user whether to apply certain fragments.

    Parameters:
    selectedKernelConfig (KernelConfig): The selected kernel configuration.
    device_choice (str): The device for which to build.

    Returns:
    list[str]: List of applied fragments in topological order.
    """
    applied_fragments = []
    defconfig_list = []
    skipped_fragments = set()  # Track fragments that were skipped

    # Filter fragments that are applicable to the current device
    applicable_fragments = [
        fragment
        for fragment in selectedKernelConfig.config_fragments
        if fragment["TargetDevice"] is None or device_choice in fragment["TargetDevice"]
    ]

    # Perform topological sorting on the applicable fragments
    sorted_fragments = topological_sort(applicable_fragments)

    # Apply the fragments in topologically sorted order
    for fragment in sorted_fragments:
        fragment_name = fragment["NamingScheme"].replace("{device}", device_choice)

        # Check if any of this fragment's dependencies were skipped
        if fragment["DependsOn"] is not None and any(
            dep in skipped_fragments for dep in fragment["DependsOn"]
        ):
            logging.info(
                f"Skipping fragment '{fragment_name}' because a dependency was not applied."
            )
            skipped_fragments.add(fragment_name)  # Mark as skipped
            continue

        # If the fragment is default-enabled, apply it automatically
        if fragment["Default"]:
            logging.info(
                f"Applying default-enabled fragment: {fragment_name} ({fragment['Description']})"
            )
            defconfig_list.append(fragment_name)
            applied_fragments.append(fragment_name)
        else:
            # Ask the user whether to apply the fragment
            # And handle the user's answer
            if ask_yesno(
                f"Apply fragment '{fragment['Description']}' ({fragment_name})?",
                allow_empty=True,
            ):
                logging.info(f"Applying user-selected fragment: {fragment_name}")
                defconfig_list.append(fragment_name)
                applied_fragments.append(fragment_name)
            else:
                logging.info(f"Skipping fragment: {fragment_name}")
                skipped_fragments.add(fragment_name)  # Mark as skipped

    return defconfig_list

def check_bins():
    """
    Check if required binaries are available in the current directory.
    """
    requiredBinaries = ["flex", "bison", "make", "git", "zip", "cpio"]
    for binary in requiredBinaries:
        if shutil.which(binary) is None:
            logging.error(f"{binary} not found in PATH")
            sys.exit(1)

def main(device: str = None, clean : bool = None, additional_defconfig_list : list[str] = None):
    currPath = Path(os.path.dirname(os.path.abspath(__file__)))
    if Path(os.getcwd()).resolve() != currPath:
        logging.error(f"Current directory is not {currPath.name}. Please chdir to it.")
        #return

    # Parse the main ini file.
    iniFile = Path() / "configs" / "kernelbuilder.ini"
    assert iniFile.is_file(), f"kernelbuilder.ini not found, cwd: {os.getcwd()}"
    mainConfig = configparser.ConfigParser()
    mainConfig.read(iniFile)

    # Load popen_impl debug settings
    debug_popen_impl = mainConfig.getboolean("popen_impl", "Debug", fallback=False)
    debug_popen_impl_writefile = mainConfig.getboolean(
        "popen_impl", "WriteLogFiles", fallback=False
    )
    if debug_popen_impl:
        if debug_popen_impl_writefile:
            popen_impl_set_debugmode(PopenImpl.DebugMode.Debug_OutputToFile)
        else:
            popen_impl_set_debugmode(PopenImpl.DebugMode.Debug)

    # Load directories config
    toolchainDirectory = Path(mainConfig.get("directory", "Toolchain"))
    OutDirectory = Path(mainConfig.get("directory", "Out"))

    # Load jobs count config
    JobCountFormula = mainConfig.get("build", "JobsCountFormula", fallback=None)
    basicMathRegex = re.compile(r"^x(\W(\+|-|\/|\*)\W\d+)?$")
    if JobCountFormula is None or not basicMathRegex.match(JobCountFormula):
        logging.warning("Invalid JobsCountFormula (Need a lvalue x and rvalue)")
        JobCountFormula = "x"
    jobsCount = eval(JobCountFormula, {"x": os.cpu_count()})
    logging.info(
        f"Calculated JobsCount: {jobsCount} from '{JobCountFormula}' where x is {os.cpu_count()}"
    )

    hostString = mainConfig.get("build", "HostString", fallback=None)
    userString = mainConfig.get("build", "UserString", fallback=None)

    # Parse the kernel specific ini files.
    kernelConfigDir = Path() / "configs" / "kernels"
    kernelConfigFiles = [
        f for f in kernelConfigDir.iterdir() if f.name.endswith(".ini")
    ]
    kernelConfigs = []
    for kernelConfigFile in kernelConfigFiles:
        try:
            kernelConfig = KernelConfig(kernelConfigFile)
        except (AssertionError, configparser.Error) as e:
            logging.error(f"Error parsing {kernelConfigFile}: {e}")
            continue
        kernelConfigs.append(kernelConfig)

    if len(kernelConfigs) == 0:
        logging.fatal("No valid kernel configurations found")
        return
    logging.info(f"Parsed {len(kernelConfigs)} kernel configurations")

    # Ask user what device to build for.
    if device is None:
        device_choices = []
        for config in kernelConfigs:
            device_choices += config.devices

        # Sort the device choices alphabetically.
        device_choices.sort()
        device_choice = choose_from_list(device_choices)
    else:
        device_choice = device
    logging.info(f"Building for device: {device_choice}")

    selectedKernelConfigList = [
        config for config in kernelConfigs if device_choice in config.devices
    ]
    if len(selectedKernelConfigList) == 0:
        logging.fatal(f"No valid kernel configurations found for device {device_choice}")
        return  
    # TODO: Maybe support multiple configs per device.
    selectedKernelConfig = selectedKernelConfigList[0]
    logging.info(f"Selected kernel: {selectedKernelConfig.name}")

    # Check if the kernel directory already exists.
    if os.path.isdir(selectedKernelConfig.simple_name()):
        logging.info(f"Kernel directory already exists, just using it.")
        os.chdir(selectedKernelConfig.simple_name())
    else:
        # Ask user if they want to clone the Kernel repo.
        kernelDirectory = input(
            """Do you want to clone the Kernel repo?
If no, provide a directory with the kernel clone, else just hit enter: """
        )
        if kernelDirectory.strip() == "":
            logging.info(f"Cloning Kernel repo from {selectedKernelConfig.repo_url}...")
            while True:
                try:
                    _input = input(
                        "Enter a clone depth in number, or press enter to clone full repository: "
                    )
                    if _input == "":
                        depth = None
                        break
                    depth = int(_input)
                    if depth < 1:
                        logging.warning("Invalid clone depth. Please enter a positive number.")
                        continue
                except ValueError:
                    logging.warning("Invalid clone depth. Please enter a number.")
                    continue
                except KeyboardInterrupt:
                    logging.warning("Interrupted. Aborting")
                    sys.exit(1)
                break
            selectedKernelConfig.clone(depth=depth)
            os.chdir(selectedKernelConfig.simple_name())
            logging.info("Done")
        else:
            logging.info(f"Using provided kernel directory: {kernelDirectory}")
            os.chdir(kernelDirectory)

    defconfig_list = [
        selectedKernelConfig.namingscheme.replace("{device}", device_choice)
    ]

    if additional_defconfig_list is None:
        # Apply fragments based on dependencies, with user interaction
        try:
            defconfig_list += apply_fragments(selectedKernelConfig, device_choice)
        except RuntimeError as e:
            logging.error(f"Error applying fragments: {str(e)}")
            return
    else:
        defconfig_list += additional_defconfig_list
        logging.info(f"Applying additional defconfigs: {additional_defconfig_list}")

    def create_toolchain_symlink():
        tc = currPath / "toolchains"

        def link(src: Path) -> bool:
            dst = toolchainDirectory
            if not src.is_dir():
                logging.warning(f"Source directory {src} does not exist")
                return False

            if dst.is_symlink():
                if os.readlink(dst) == src:
                    logging.info(f"Symlink {src} => {dst} already exists")
                    return True
                else:
                    logging.info(f"Removing existing symlink {dst}")
                    os.remove(dst)

            try:
                os.symlink(src, dst)
                logging.info(f"Created link {src} => {dst}")
                return True
            except OSError as e:
                logging.warning(f"Failed to create link {src} => {dst}: {e}")
                return False

        match selectedKernelConfig.toolchain_config:
            case ToolchainConfig.GCCOnly:
                for i in ["gcc-android-", "gcc-"]:
                    done = link(tc / f"{i}{selectedKernelConfig.kernel_arch.to_str()}")
                    if done:
                        break
            # Others use clang
            case _:
                done = link(tc / "clang")

        if not done:
            logging.warning("Failed to create toolchain directory or link toolchain")
            return False
        return True

    if not toolchainDirectory.is_dir() and not toolchainDirectory.is_symlink():
        logging.warning(f"Toolchain directory does not exist")
        if not create_toolchain_symlink():
            return
    if (
        toolchainDirectory.is_symlink()
        and not Path(os.readlink(toolchainDirectory)).is_dir()
    ):
        logging.warning(f"Toolchain directory points to a non-existing directory")
        if not create_toolchain_symlink():
            return

    try:
        compilerType, arglist, targetTriple = find_available_compilers(
            toolchainDirectory,
            selectedKernelConfig.kernel_arch,
            selectedKernelConfig.toolchain_config,
        )
        logging.info(
            f"toolchain version: {compilerType.toolchain_version(toolchainDirectory / 'bin', selectedKernelConfig.kernel_arch)}"
        )
    except UnImplementedError as e:
        logging.error(f"Error finding available compilers: {str(e)}")
        return

    # Init submodules, if any.
    if os.path.isfile(".gitmodules"):
        try:
            popen_impl(["git", "submodule", "update", "--init"])
            logging.info("Submodules initialized successfully")
        except RuntimeError:
            logging.error("Error initializing submodules")
            return

    arch = selectedKernelConfig.kernel_arch.to_str()
    type = selectedKernelConfig.kernel_type.to_str()

    # Add toolchain in PATH environment variable
    tcPath = Path(os.getcwd()) / toolchainDirectory / "bin"
    newEnv = os.environ.copy()
    newEnv["PATH"] = tcPath.as_posix() + ":" + newEnv["PATH"]
    if selectedKernelConfig.envMap:
        for key, value in selectedKernelConfig.envMap.items():
            logging.debug(f"Setting ENV {key} to {value}...")
            newEnv[key] = value

    # Append custom strings in the make command
    if hostString:
        newEnv["KBUILD_BUILD_HOST"] = hostString
    if userString:
        newEnv["KBUILD_BUILD_USER"] = userString

    # Clean the Out directory if it exists, but ask before.
    def cleanFn():
        logging.info("Make clean...")
        shutil.rmtree(OutDirectory)
        os.mkdir(OutDirectory)

    if clean is None:
        if OutDirectory.is_dir() and ask_yesno("Clean the Out directory?"):
            cleanFn()
    elif clean == True:
        cleanFn()

    make_defconfig = []
    make_kernel = []
    make_common = [
        "make",
        "ARCH=" + arch,
        "O=" + OutDirectory.as_posix(),
        "CROSS_COMPILE=" + targetTriple,
        "CROSS_COMPILE_ARM32=arm-linux-gnueabi",
        f"-j{jobsCount}",
    ]
    make_common += arglist
    make_defconfig += make_common
    make_defconfig += defconfig_list
    make_kernel += make_common
    make_kernel += [type]

    t = datetime.now()
    try:
        logging.info("Make defconfig...")
        popen_impl(make_defconfig, env=newEnv, ext_diag=True)
        logging.info("Make kernel...")
        popen_impl(make_kernel, env=newEnv, timed_output=True, ext_diag=True)
        logging.info("Done")
    except RuntimeError as e:
        # If these failed, then goodbye.
        logging.error(str(e))
        return
    t = datetime.now() - t

    with open(OutDirectory / "include" / "generated" / "utsrelease.h") as f:
        kver = match_and_get_first(r'"([^"]+)"', f.read())

    newZipName = Path()
    if selectedKernelConfig.anykernel3_directory:
        AnyKernelDirectory = selectedKernelConfig.anykernel3_directory

        shutil.copyfile(
            OutDirectory / "arch" / arch / "boot" / type,
            AnyKernelDirectory / type,
        )
        zipname = (selectedKernelConfig.simple_name() + "_{}_{}.zip").format(
            device_choice, datetime.today().strftime("%Y-%m-%d")
        )
        os.chdir(AnyKernelDirectory)
        zippinglist = [
            type,
            "META-INF/com/google/android/update-binary",
            "META-INF/com/google/android/updater-script",
            "tools/ak3-core.sh",
            "tools/busybox",
            "tools/magiskboot",
            "anykernel.sh",
        ]
        if len(selectedKernelConfig.anykernel3_addfiles) != 0:
            zippinglist += selectedKernelConfig.anykernel3_addfiles
            logging.info(
                f"Adding additional files: {[str(x) for x in selectedKernelConfig.anykernel3_addfiles]}"
            )
        zip_files(zipname, zippinglist)
        newZipName = Path(os.getcwd()) / ".." / zipname
        try:
            newZipName.unlink()
        except:
            pass
        shutil.move(zipname, newZipName)
        os.chdir("..")
        logging.info("Kernel zip created: " + str(newZipName.resolve()))
    else:
        logging.info("Skipping AnyKernel3 zip creation")
    logging.info("Kernel version: " + kver)
    logging.info("Elapsed time: " + str(t.total_seconds()) + " seconds")
    os.chdir(currPath)
    return newZipName

if __name__ == "__main__":
    check_bins()
    main()
