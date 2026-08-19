#include "engine/core/ui/file_dialog.hpp"

#include <ImGuiFileDialog.h>

namespace KE {
namespace {

ImGuiFileDialogFlags commonFlags() {
    return ImGuiFileDialogFlags_Modal |
           ImGuiFileDialogFlags_DontShowHiddenFiles |
           ImGuiFileDialogFlags_DisableThumbnailMode;
}

IGFD::FileDialogConfig configFor(const std::string& path,
                                 const std::string& fileName = {}) {
    IGFD::FileDialogConfig config;
    config.path = path.empty() ? "." : path;
    config.fileName = fileName;
    config.countSelectionMax = 1;
    config.flags = commonFlags();
    return config;
}

} // namespace

FileDialogService& FileDialogService::instance() {
    static FileDialogService service;
    return service;
}

void FileDialogService::openFile(const std::string& key,
                                 const std::string& title,
                                 const std::string& filters,
                                 const std::string& path) {
    _directoryKeys.erase(key);
    ImGuiFileDialog::Instance()->OpenDialog(
        key, title, filters.empty() ? nullptr : filters.c_str(),
        configFor(path));
}

void FileDialogService::openSave(const std::string& key,
                                 const std::string& title,
                                 const std::string& filters,
                                 const std::string& path,
                                 const std::string& fileName) {
    _directoryKeys.erase(key);
    auto config = configFor(path, fileName);
    config.flags |= ImGuiFileDialogFlags_ConfirmOverwrite;
    ImGuiFileDialog::Instance()->OpenDialog(
        key, title, filters.empty() ? nullptr : filters.c_str(), config);
}

void FileDialogService::openDirectory(const std::string& key,
                                      const std::string& title,
                                      const std::string& path) {
    _directoryKeys.insert(key);
    ImGuiFileDialog::Instance()->OpenDialog(key, title, nullptr,
                                            configFor(path));
}

FileDialogResult FileDialogService::display(const std::string& key) {
    FileDialogResult result;
    auto* dialog = ImGuiFileDialog::Instance();
    if (!dialog->Display(key))
        return result;

    result.finished = true;
    if (dialog->IsOk()) {
        result.path = _directoryKeys.count(key) != 0
                          ? dialog->GetCurrentPath()
                          : dialog->GetFilePathName();
    }
    _directoryKeys.erase(key);
    dialog->Close();
    return result;
}

void FileDialogService::close() { ImGuiFileDialog::Instance()->Close(); }

} // namespace KE
