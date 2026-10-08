// SdhcalPaths.h
//
// Data locations for the C++ and ROOT side, resolved at run time instead of
// being written into every macro. The Python side does the same through
// sdhcal/paths.py; the two agree on the same environment variables and the same
// directory layout.
//
//     SDHCAL_DATA_DIR    root of the ROOT inputs   (default: <repo>/data)
//
// Usage from a macro:
//
//     #include "../../cpp/include/SdhcalPaths.h"   // from analysis/<stage>/
//     const std::string f = sdhcal::digitized("pi");
//
// The expected layout under SDHCAL_DATA_DIR:
//
//     params/130k_<particle>_E1to130_params.root
//     merged_primaryEnergy/130k_<particle>_E1to130_params_merged.root
//     digitized/130k_<particle>_E1to130_digitized.root
//     raw/130k_<particle>_E1to130.root
//     val_set.root

#ifndef SDHCAL_PATHS_H
#define SDHCAL_PATHS_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>

namespace sdhcal {

/// True when the file exists and is readable.
inline bool exists(const std::string& path) {
  struct stat buffer;
  return stat(path.c_str(), &buffer) == 0;
}

/// Repository root, found by walking up from the current directory until a
/// marker file appears.
///
/// A ROOT macro may be run from the repository root, from its own stage
/// directory, or from anywhere else, so a relative default such as "../data"
/// would resolve differently each time. Looking for the marker makes the result
/// independent of the working directory.
inline std::string repoRoot() {
  std::string prefix;
  for (int level = 0; level < 8; ++level) {
    if (exists(prefix + "pyproject.toml") && exists(prefix + "src/sdhcal")) {
      return prefix.empty() ? std::string(".") : prefix.substr(0, prefix.size() - 1);
    }
    prefix += "../";
  }
  return std::string(".");
}

/// Root of the ROOT inputs: $SDHCAL_DATA_DIR, or <repo>/data.
inline std::string dataDir() {
  const char* fromEnv = std::getenv("SDHCAL_DATA_DIR");
  if (fromEnv && *fromEnv) return std::string(fromEnv);
  return repoRoot() + "/data";
}

/// Where run outputs go: $SDHCAL_OUT_DIR, or <repo>/output. Same convention as
/// sdhcal.paths.output_dir() on the Python side.
inline std::string outputDir() {
  const char* fromEnv = std::getenv("SDHCAL_OUT_DIR");
  if (fromEnv && *fromEnv) return std::string(fromEnv);
  return repoRoot() + "/output";
}

/// Directory of the shared chi2 calibration file.
inline std::string calibrationDir() { return repoRoot() + "/calibration"; }

/// Warn, once per call, when a path is missing, naming the variable to set.
inline const std::string& warnIfMissing(const std::string& path) {
  if (!exists(path)) {
    std::cerr << "WARNING: missing input " << path
              << "\n         set SDHCAL_DATA_DIR to the directory holding the ROOT files\n";
  }
  return path;
}

/// Any path relative to the data directory, for files whose name does not follow
/// the 130k_<particle>_E1to130 convention (the 200k comparison samples, say).
inline std::string inDataDir(const std::string& relative) {
  return warnIfMissing(dataDir() + "/" + relative);
}

inline std::string params(const std::string& particle) {
  return warnIfMissing(dataDir() + "/params/130k_" + particle + "_E1to130_params.root");
}

inline std::string paramsMerged(const std::string& particle) {
  return warnIfMissing(dataDir() + "/merged_primaryEnergy/130k_" + particle +
                       "_E1to130_params_merged.root");
}

inline std::string digitized(const std::string& particle) {
  return warnIfMissing(dataDir() + "/digitized/130k_" + particle + "_E1to130_digitized.root");
}

inline std::string raw(const std::string& particle) {
  return warnIfMissing(dataDir() + "/raw/130k_" + particle + "_E1to130.root");
}

inline std::string holdout() { return warnIfMissing(dataDir() + "/val_set.root"); }

}  // namespace sdhcal

#endif  // SDHCAL_PATHS_H
