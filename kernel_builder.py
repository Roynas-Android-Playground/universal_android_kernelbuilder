import configparser
import subprocess
import os
import shutil
import sys
import re
import logging
import zipfile
from datetime import datetime
from enum import Enum

# Set default encoding to UTF-8
UTF8Codec = "utf-8"

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(filename)s:%(lineno)-3d - %(levelname)-8s: %(message)s",
)


class PopenImpl:
    class DebugMode(Enum):
        Off = (0,)
        Debug = (1,)
        Debug_OutputToStderr = (2,)
        Debug_OutputToFile = (3,)

        def isDebug(self) -> bool:
            return self != self.Off

    debugmode = DebugMode.Off

    def run(self, command: list[str]):
        """
        Execute a command using subprocess.Popen and handle its output and errors.

        Parameters:
        command (list[str]): The command to be executed.

        Raises:
        RuntimeError: If the command execution fails.
        OSError: subprocess.Popen would throw that additionally.

        Returns:
        None
        """
        if self.debugmode.isDebug():
            logging.debug("Executing command: %s", " ".join(command))

        s = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = s.communicate()

        def write_logs(out, err, failed=False):
            out = out.decode(UTF8Codec)
            err = err.decode(UTF8Codec)
            if self.debugmode == self.DebugMode.Debug_OutputToStderr and failed:
                logging.error("Failed, printing process stderr output to stderr...")
                print(err, file=sys.stderr)
            elif self.debugmode == self.DebugMode.Debug_OutputToFile:
                stdout_log = str(s.pid) + "_stdout.log"
                stderr_log = str(s.pid) + "_stderr.log"
                with open(stdout_log, "w") as f:
                    f.write(out)
                with open(stderr_log, "w") as f:
                    f.write(err)
                logging.info(f"Output log files: {stdout_log}, {stderr_log}")

        if s.returncode != 0:
            write_logs(out, err, failed=True)
            raise RuntimeError(f"Command failed: {command}. Exitcode: {s.returncode}")
        if self.debugmode.isDebug():
            logging.debug(f"result: {s.returncode == 0}")
            write_logs(out, err)

    def set_debugmode(self, mode: DebugMode) -> None:
        self.debugmode = mode


_popen_impl_inst = PopenImpl()
popen_impl = _popen_impl_inst.run
popen_impl_set_debugmode = _popen_impl_inst.set_debugmode


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


class CompilerType(Enum):
    GCC = "gcc"
    Clang = "clang"


class CompilerTest:
    def __init__(self, exe_path: str, versionRegex: str) -> None:
        """
        Initialize a CompilerTest instance.

        Parameters:
        exe_path (str): The path to the compiler executable.
        versionRegex (str): The regular expression used to extract the version from the compiler's output.

        Returns:
        None
        """
        self.versionRegex = versionRegex
        self.versionArgList = [exe_path, "--version"]
        self.executable_path = exe_path

    def test_executable(self):
        """
        Test if the compiler executable can be executed.

        Parameters:
        None

        Returns:
        bool: True if the executable can be executed, False otherwise.
        """
        try:
            popen_impl(self.versionArgList)
            return True
        except (OSError, RuntimeError) as e:
            logging.error(f"Failed to execute: {e}")
            return False

    def get_version(self):
        """
        Get the version of the compiler.

        This function executes the compiler with the '--version' argument and extracts the version
        information using a regular expression.

        Parameters:
        None

        Returns:
        str: The version of the compiler.
        """
        s = subprocess.Popen(
            self.versionArgList, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        tcversion, _ = s.communicate()
        tcversion = tcversion.decode(UTF8Codec)
        return match_and_get_first(self.versionRegex, tcversion)

    def get_path(self) -> str:
        """
        Get the path of the compiler executable.

        Returns:
        str: The path of the compiler executable.
        """
        return self.executable_path

    def get_compiler_type(self) -> CompilerType:
        """
        Determine the type of compiler based on the executable path.

        Returns:
        CompilerType: The type of compiler (either GCC or Clang).

        Raises:
        ValueError: If the compiler executable path does not match either GCC or Clang.
        """
        if self.executable_path.endswith("gcc"):
            return CompilerType.GCC
        elif self.executable_path.endswith("clang"):
            return CompilerType.Clang
        else:
            raise ValueError("Unsupported compiler executable path")


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


class ToolchainConfig(Enum):
    GCCOnly = ("GCCOnly",)
    GNUBinClang = ("GNUBinClang",)
    FullLLVM = "FullLLVM"

    @classmethod
    def from_str(cls, toolchain_str: str) -> "ToolchainConfig":
        try:
            return cls[toolchain_str]
        except KeyError:
            raise ValueError(f"Unsupported toolchain configuration: {toolchain_str}")

    def to_str(self):
        return self.name


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

    def __init__(self, config_file: str) -> None:
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
        assert os.path.isfile(
            self.config_file
        ), f"Kernelconfig file not found: {self.config_file}"
        self.config.read(self.config_file)
        self._parse_info()
        self._parse_defconfig()
        self._parse_config_fragments()
        logging.debug(f"Config parsed successfully: {self.name}")

    def clone(self):
        try:
            popen_impl(
                [
                    "git",
                    "clone",
                    self.repo_url,
                    "--depth=1",
                    "-b",
                    self.repo_branch,
                    self._simple_name,
                ]
            )
        except RuntimeError as e:
            logging.error(f"Failed to clone repository: {e}")
            raise

    def simple_name(self) -> str:
        return self._simple_name


class UnImplementedError(Exception):
    pass


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
    logging.info(
        f"Zipping {len(files)} files to {os.path.relpath(os.getcwd(), zipfilename)}..."
    )
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

def find_available_compilers(toolchain_directory, kernel_arch, toolchain_config):
    """
    Find available compilers based on the kernel architecture and toolchain configuration.

    Parameters:
    - toolchain_directory (str): The path to the toolchain directory.
    - kernel_arch (KernelArch): The architecture of the kernel.
    - toolchain_config (ToolchainConfig): The toolchain configuration (GCCOnly, FullLLVM, etc.).

    Returns:
    - list[CompilerTest]: A list of available compiler tests.
    """
    gcc_prefixes = {
        KernelArch.x86_64: ["x86_64-linux-gnu-", "x86_64-linux-android-"],
        KernelArch.arm64: ["aarch64-linux-gnu-", "aarch64-linux-android-"],
        KernelArch.arm: ["arm-linux-gnueabi-", "arm-linux-androideabi-"],
        KernelArch.x86: ["i686-linux-gnu-", "i686-linux-android-"],
    }

    clang_prefixes = {
        KernelArch.x86_64: "x86_64-linux-gnu-",
        KernelArch.arm64: "aarch64-linux-gnu-",
        KernelArch.arm: "arm-linux-gnueabi-",
        KernelArch.x86: "i686-linux-gnu-",
    }

    compilers = []

    # Check the toolchain configuration and create appropriate CompilerTest objects
    if toolchain_config == ToolchainConfig.FullLLVM:
        clang_compiler = CompilerTest(
            exe_path=os.path.join(toolchain_directory, "bin", "clang"),
            versionRegex=r"(.*?clang version \d+(\.\d+)*).*"
        )
        compilers.append(clang_compiler)

    for prefix in gcc_prefixes[kernel_arch]:
        gcc_compiler = CompilerTest(
            exe_path=os.path.join(toolchain_directory, "bin", prefix + "gcc"),
            versionRegex=r"(.*?gcc \(.*\) \d+(\.\d+)*)"
        )
        compilers.append(gcc_compiler)

    return compilers


def select_compiler(compilers):
    """
    Select the first available compiler by testing each one in order.

    Parameters:
    - compilers (list[CompilerTest]): A list of CompilerTest objects to test.

    Returns:
    - CompilerTest: The first available compiler, or None if no compiler is found.
    """
    for compiler in compilers:
        if compiler.test_executable():
            logging.info(f"Selected compiler: {compiler.get_version()}")
            return compiler
        else:
            logging.warning(f"Compiler {compiler.get_path()} not available.")
    return None


def determine_target_triple(compiler, kernel_arch):
    """
    Determine the target triple based on the selected compiler and kernel architecture.

    Parameters:
    - compiler (CompilerTest): The selected compiler.
    - kernel_arch (KernelArch): The architecture of the kernel.

    Returns:
    - str: The target triple for cross-compilation, or None if unsupported.
    """
    clang_prefixes = {
        KernelArch.x86_64: "x86_64-linux-gnu-",
        KernelArch.arm64: "aarch64-linux-gnu-",
        KernelArch.arm: "arm-linux-gnueabi-",
        KernelArch.x86: "i686-linux-gnu-",
    }

    if compiler.get_compiler_type() == CompilerType.Clang:
        return clang_prefixes[kernel_arch]
    elif compiler.get_compiler_type() == CompilerType.GCC:
        return os.path.basename(compiler.get_path())[:-3]  # Strip 'gcc' suffix
    else:
        logging.error("Unsupported compiler type.")
        return None


def choose_compiler(toolchain_directory, toolchain_config, kernel_arch):
    """
    Choose the appropriate compiler for the build process based on the toolchain configuration
    and kernel architecture.

    Parameters:
    - toolchain_directory (str): The path to the toolchain directory.
    - toolchain_config (ToolchainConfig): The toolchain configuration (e.g., GCCOnly, FullLLVM).
    - kernel_arch (KernelArch): The architecture of the kernel (e.g., x86_64, arm64).

    Returns:
    - tuple[CompilerTest, str]: The selected compiler and its target triple, or (None, None) if no compiler is found.
    """
    logging.info("Searching for available compilers...")

    try:
        compilers = find_available_compilers(toolchain_directory, kernel_arch, toolchain_config)

        selected_compiler = select_compiler(compilers)
        if not selected_compiler:
            logging.error("No suitable compiler found.")
            return None, None

        target_triple = determine_target_triple(selected_compiler, kernel_arch)
        if not target_triple:
            logging.error("Unable to determine target triple.")
            return None, None

        return selected_compiler, target_triple

    except UnImplementedError as e:
        logging.error(f"Unimplemented toolchain configuration: {str(e)}")
        return None, None

llvm_sets = {
    "CC": "clang",
    "LD": "ld.lld",
    "AR": "llvm-ar",
    "NM": "llvm-nm",
    "OBJCOPY": "llvm-objcopy",
    "OBJDUMP": "llvm-objdump",
}

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
    in_degree = defaultdict(int)  # Track in-degree of each fragment (number of dependencies)

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
        current_fragment = queue.popleft()  # Get a fragment with no remaining dependencies
        sorted_fragments.append(current_fragment)

        # Reduce the in-degree of dependent fragments
        for dependent in graph[current_fragment["NamingScheme"]]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:  # If no more dependencies, add to queue
                queue.append(next(frag for frag in fragments if frag["NamingScheme"] == dependent))

    # Check if there's a cycle (unresolved dependencies)
    if len(sorted_fragments) != len(fragments):
        raise RuntimeError("Dependency cycle detected in fragments")

    return sorted_fragments

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
        fragment for fragment in selectedKernelConfig.config_fragments
        if fragment["TargetDevice"] is None or device_choice in fragment["TargetDevice"]
    ]

    # Perform topological sorting on the applicable fragments
    sorted_fragments = topological_sort(applicable_fragments)

    # Apply the fragments in topologically sorted order
    for fragment in sorted_fragments:
        fragment_name = fragment["NamingScheme"].replace("{device}", device_choice)

        # Check if any of this fragment's dependencies were skipped
        if fragment["DependsOn"] is not None and any(dep in skipped_fragments for dep in fragment["DependsOn"]):
            logging.info(f"Skipping fragment '{fragment_name}' because a dependency was not applied.")
            skipped_fragments.add(fragment_name)  # Mark as skipped
            continue

        # If the fragment is default-enabled, apply it automatically
        if fragment["Default"]:
            logging.info(f"Applying default-enabled fragment: {fragment_name} ({fragment['Description']})")
            defconfig_list.append(fragment_name)
            applied_fragments.append(fragment_name)
        else:
            # Ask the user whether to apply the fragment
            valid_answers = ["y", "n", ""]
            yes_answers = ["y", ""]
            answer = input(f"Apply fragment '{fragment['Description']}' ({fragment_name})? (Y/n): ").lower()

            # Handle the user's answer
            if answer in valid_answers:
                if answer in yes_answers:
                    logging.info(f"Applying user-selected fragment: {fragment_name}")
                    defconfig_list.append(fragment_name)
                    applied_fragments.append(fragment_name)
                else:
                    logging.info(f"Skipping fragment: {fragment_name}")
                    skipped_fragments.add(fragment_name)  # Mark as skipped
            else:
                logging.warning(f"Invalid input. Skipping fragment: {fragment_name}")
                skipped_fragments.add(fragment_name)  # Mark as skipped

    return defconfig_list



def main():
    # Parse the main ini file.
    iniFile = os.path.join("configs", "kernelbuilder.ini")
    assert os.path.isfile(iniFile), "kernelbuilder.ini not found"
    mainConfig = configparser.ConfigParser()
    mainConfig.read(iniFile)

    # Load popen_impl debug settings
    debug_popen_impl = mainConfig.getboolean("popen_impl", "Debug", fallback=False)
    debug_popen_impl_writefile = mainConfig.getboolean(
        "popen_impl", "WriteLogFiles", fallback=False
    )
    debug_popen_impl_showstderr = mainConfig.getboolean(
        "popen_impl", "ShowStdErrToConsole", fallback=False
    )
    if debug_popen_impl:
        if debug_popen_impl_showstderr:
            popen_impl_set_debugmode(PopenImpl.DebugMode.Debug_OutputToStderr)
        elif debug_popen_impl_writefile:
            popen_impl_set_debugmode(PopenImpl.DebugMode.Debug_OutputToFile)
        else:
            popen_impl_set_debugmode(PopenImpl.DebugMode.Debug)

    # Load directories config
    toolchainDirectory = mainConfig.get("directory", "Toolchain")
    AnyKernelDirectory = mainConfig.get("directory", "AnyKernel")
    OutDirectory = mainConfig.get("directory", "Out")

    # Load jobs count config
    JobCountFormula = mainConfig.get("build", "JobsCountFormula", fallback="x")
    basicMathRegex = re.compile(r"^x(\W(\+|-|\/|\*)\W\d+)?$")
    if not basicMathRegex.match(JobCountFormula):
        logging.warning("Invalid JobsCountFormula (Need a lvalue x and rvalue)")
        JobCountFormula = "x"
    jobsCount = eval(JobCountFormula, {"x": os.cpu_count()})
    logging.info(
        f"Calculated JobsCount: {jobsCount} from '{JobCountFormula}' where x is {os.cpu_count()}"
    )

    # Parse the kernel specific ini files.
    kernelConfigDir = os.path.join("configs", "kernels", "")
    kernelConfigFiles = [f for f in os.listdir(kernelConfigDir) if f.endswith(".ini")]
    kernelConfigs = []
    for kernelConfigFile in kernelConfigFiles:
        try:
            kernelConfig = KernelConfig(os.path.join(kernelConfigDir, kernelConfigFile))
        except (AssertionError, configparser.Error) as e:
            logging.error(f"Error parsing {kernelConfigFile}: {e}")
            continue
        kernelConfigs.append(kernelConfig)

    if len(kernelConfigs) == 0:
        logging.fatal("No valid kernel configurations found")
        return
    logging.info(f"Parsed {len(kernelConfigs)} kernel configurations")

    # Ask user what device to build for.
    device_choices = []
    for config in kernelConfigs:
        device_choices += config.devices

    # Sort the device choices alphabetically.
    device_choices.sort()
    device_choice = choose_from_list(device_choices)
    logging.info(f"Building for device: {device_choice}")

    # TODO: Maybe support multiple configs per device.
    selectedKernelConfig = [
        config for config in kernelConfigs if device_choice in config.devices
    ][0]
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
            selectedKernelConfig.clone()
            os.chdir(selectedKernelConfig.simple_name())
            logging.info("Done")
        else:
            logging.info(f"Using provided kernel directory: {kernelDirectory}")
            os.chdir(kernelDirectory)

    defconfig_list = [
        selectedKernelConfig.namingscheme.replace("{device}", device_choice)
    ] 
    
    # Apply fragments based on dependencies, with user interaction
    try:
        defconfig_list += apply_fragments(selectedKernelConfig, device_choice)
    except RuntimeError as e:
        logging.error(f"Error applying fragments: {str(e)}")
        return

    selectedCompiler, targetTriple = choose_compiler(
        toolchainDirectory,
        selectedKernelConfig.toolchain_config,
        selectedKernelConfig.kernel_arch,
    )

    if targetTriple is None:
        return

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
    os.mkdir(OutDirectory)

    make_defconfig = []
    make_common = [
        "make",
        "ARCH=" + arch,
        "O=" + OutDirectory,
        "CROSS_COMPILE=" + targetTriple,
        f"-j{jobsCount}",
    ]
    if selectedCompiler.get_compiler_type() == CompilerType.Clang:
        make_common += [f"{x}={y}" for x, y in llvm_sets.items()]
    make_defconfig += make_common
    make_defconfig += defconfig_list

    # Check for some binaries that are required in the kernel build.
    for bins in ["flex", "bison", "make", "git", "zip"]:
        if not shutil.which(bins):
            logging.error(f"{bins} not found in PATH")
            return

    t = datetime.now()
    try:
        logging.info("Make defconfig...")
        popen_impl(make_defconfig)
        logging.info("Make kernel...")
        popen_impl(make_common)
        logging.info("Done")
    except RuntimeError as e:
        # If these failed, then goodbye.
        logging.error(str(e))
        return
    t = datetime.now() - t

    with open(os.path.join(OutDirectory, "include", "generated", "utsrelease.h")) as f:
        kver = match_and_get_first(r'"([^"]+)"', f.read())

    shutil.copyfile(
        os.path.join(OutDirectory, "arch", arch, "boot", type),
        os.path.join(AnyKernelDirectory, type),
    )
    zipname = (selectedKernelConfig.dest_dir() + "_{}_{}.zip").format(
        device_choice, datetime.today().strftime("%Y-%m-%d")
    )
    os.chdir(AnyKernelDirectory)
    zip_files(
        zipname,
        [
            type,
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
