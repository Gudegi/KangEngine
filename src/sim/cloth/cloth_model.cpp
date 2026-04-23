#include "cloth_model.hpp"
#include <cmath>

namespace KE {
namespace Sim {

ClothModel ClothModel::createGrid(int rows, int cols, float width, float height) {
    ClothModel m;
    m.rows = rows;
    m.cols = cols;
    m.width = width;
    m.height = height;

    float dx = width / (cols - 1);
    float dy = height / (rows - 1);

    // UV
    m.uv.resize(rows * cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            m.uv[m.index(r, c)] = {(float)(c) / (cols - 1),
                                   (float)(r) / (rows - 1)};

    // Triangle indices (two triangles per quad)
    for (int r = 0; r < rows - 1; r++) {
        for (int c = 0; c < cols - 1; c++) {
            uint32_t tl = m.index(r, c);
            uint32_t tr = m.index(r, c + 1);
            uint32_t bl = m.index(r + 1, c);
            uint32_t br = m.index(r + 1, c + 1);
            m.indices.insert(m.indices.end(), {tl, bl, br, tl, br, tr});
        }
    }

    // Helper: add constraint with rest length from grid spacing
    auto addC = [&](std::vector<Constraint>& list, int partA, int partB,
                    float len) { list.push_back({partA, partB, len}); };

    float diagLen = std::sqrt(dx * dx + dy * dy);
    float skipH = dx * 2.f;
    float skipV = dy * 2.f;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int cur = m.index(r, c);

            // Stretch — horizontal
            if (c + 1 < cols)
                addC(m.stretches, cur, m.index(r, c + 1), dx);
            // Stretch — vertical
            if (r + 1 < rows)
                addC(m.stretches, cur, m.index(r + 1, c), dy);

            // Shear — both diagonals
            if (r + 1 < rows && c + 1 < cols)
                addC(m.shears, cur, m.index(r + 1, c + 1), diagLen);
            if (r + 1 < rows && c - 1 >= 0)
                addC(m.shears, cur, m.index(r + 1, c - 1), diagLen);

            // Bend — skip-1 horizontal
            if (c + 2 < cols)
                addC(m.bends, cur, m.index(r, c + 2), skipH);
            // Bend — skip-1 vertical
            if (r + 2 < rows)
                addC(m.bends, cur, m.index(r + 2, c), skipV);
        }
    }

    return m;
}

} // namespace Sim
} // namespace KE
