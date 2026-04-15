#ifndef _TOKEN_HPP_
#define _TOKEN_HPP_

#include <string>
#include <unordered_map>
#include <vector>
#include <cstddef>

namespace KE {
namespace Scene {

class Token {
  private:
    size_t _id;

    // 모든 Token 객체가 공유하는 테이블
    static std::unordered_map<std::string, size_t>& getTokenTable() {
        static std::unordered_map<std::string, size_t> tokenTable;
        return tokenTable;
    }

    static std::unordered_map<size_t, std::string>& getReverseTable() {
        static std::unordered_map<size_t, std::string> reverseTable;
        return reverseTable;
    }

    static size_t getOrCreateId(const std::string& str) {
        static size_t nextId = 1;
        auto& tokenTable = getTokenTable();
        auto& reverseTable = getReverseTable();

        auto it = tokenTable.find(str);
        if (it != tokenTable.end()) {
            return it->second;
        }

        tokenTable[str] = nextId;
        reverseTable[nextId] = str;
        return nextId++;
    }

  public:
    Token() : _id(0) {}
    explicit Token(const std::string& str) { _id = getOrCreateId(str); }

    // comparison
    bool operator==(const Token& other) const { return _id == other._id; }
    bool operator!=(const Token& other) const { return _id != other._id; }

    size_t id() const { return _id; }

    // to find original string
    const std::string& str() const {
        static const std::string empty;
        auto& reverseTable = getReverseTable();
        auto it = reverseTable.find(_id);
        return (it != reverseTable.end()) ? it->second : empty;
    }

    // unordered_map hash
    struct Hash {
        size_t operator()(const Token& t) const noexcept { return t._id; }
    };
};

} // namespace Scene
} // namespace KE

#endif