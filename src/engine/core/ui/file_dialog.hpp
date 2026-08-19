#pragma once

#include <optional>
#include <string>
#include <unordered_set>

namespace KE {

struct FileDialogResult {
    bool finished = false;
    std::optional<std::string> path;
};

class FileDialogService {
  public:
    static FileDialogService& instance();

    void openFile(const std::string& key, const std::string& title,
                  const std::string& filters, const std::string& path);
    void openSave(const std::string& key, const std::string& title,
                  const std::string& filters, const std::string& path,
                  const std::string& fileName);
    void openDirectory(const std::string& key, const std::string& title,
                       const std::string& path);
    FileDialogResult display(const std::string& key);
    void close();

  private:
    std::unordered_set<std::string> _directoryKeys;
};

} // namespace KE
