//
#include <array>
#include <random>
#include <tuple>
#include <vector>
//
#include <tbb/parallel_for.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <Eigen/Dense>
#include <glm/glm.hpp>

namespace detail {
template <typename Type>
class range_impl {
  public:
    struct iterator {
        inline iterator(Type v) : v_(v) {}
        inline Type operator*() { return v_; }
        inline bool operator!=(const iterator other) { return v_ != other.v_; }
        inline iterator& operator++()
        {
            v_++;
            return *this;
        }
        Type v_;
    };
    range_impl(Type b, Type e) : begin_(b), end_(e) {}
    inline iterator begin() const { return begin_; }
    inline iterator end() const { return end_; }
    inline Type size() const { return end_ - begin_; }

  private:
    const Type begin_;
    const Type end_;
};
}  // namespace detail
using Range = detail::range_impl<int32_t>;

template <typename Type>
inline detail::range_impl<Type> range(Type v)
{
    return detail::range_impl<Type>(0, v);
}

template <typename Type>
inline detail::range_impl<Type> range(Type b, Type e)
{
    return detail::range_impl<Type>(b, e);
}

// parallel for
template <typename Type, typename Fn, bool USE_TBB = true>
void pfor(const Type N, const Fn&& fn)
{
    tbb::auto_partitioner ap;
    tbb::parallel_for(
        tbb::blocked_range<Type>(0, N),
        [&](const tbb::blocked_range<Type>& r) {
            fn(range(r.begin(), r.end()));
        },
        ap);
}

class Image {
  public:
    Image(const char* filename)
    {
        float* data = stbi_loadf(filename, &width_, &height_, &comp_, 0);
        pixel_.reserve(width_ * height_);
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                const float* ptr = data + (x + y * width_) * 3;
                pixel_.push_back(glm::vec3(ptr[0], ptr[1], ptr[2]));
            }
        }
        stbi_image_free(data);
    }
    Image(int width, int height)
        : width_(width), height_(height)
    {
        pixel_.resize(width * height);
    }
    void save(const char* filename)
    {
        stbi_write_hdr(filename, width_, height_, 3, (const float*)pixel_.data());
    }
    glm::vec3& operator()(int x, int y)
    {
        return pixel_[x + y * width_];
    }
    int32_t width() const
    {
        return width_;
    }
    int32_t height() const
    {
        return height_;
    }

  private:
    std::vector<glm::vec3> pixel_;
    int32_t width_;
    int32_t height_;
    int32_t comp_;
};

const auto toY = [](glm::vec3 color) {
    return glm::dot(color, glm::vec3(0.299f, 0.587f, 0.114f));
};

//
int main(int argc, char* argv[])
{
    //
    if (argc != 5) {
        return -1;
    }
    // color.hdr albedo.hdr normal.hdr denoised.hdr
    const char* colorFileName = argv[1];
    const char* albedoFileName = argv[2];
    const char* normalFileName = argv[3];
    const char* denoisedFileName = argv[4];
    Image imgColor(colorFileName);
    Image imgAlbedo(albedoFileName);
    Image imgNormal(normalFileName);
    Image dstImg(imgColor.width(), imgColor.height());

    std::atomic<int> lineCount = 0;
    printf("[");
    pfor(imgColor.height(), [&](const auto& yr) {
        // カーネルサイズ
        constexpr auto kernelSize = 11;
        constexpr auto kernelSizeSqr = kernelSize * kernelSize;
        constexpr auto kernelSizeHalf = kernelSize / 2;
        // 座標2次元+アルベド3次元+法線3次元+bias の9次元の説明変数の行列
        Eigen::MatrixXf A = Eigen::MatrixXf::Zero(kernelSizeSqr, 9);
        // カラー3次元の目的変数の行列
        Eigen::MatrixXf b = Eigen::MatrixXf::Zero(kernelSizeSqr, 3);
        //
        for (const auto y : yr) {
            for (const auto x : range(0, imgColor.width())) {
                A.setZero();
                b.setZero();
                //
                const int uy = std::max(y - kernelSizeHalf, 0);
                const int dy = std::min(y + kernelSizeHalf, imgColor.height() - 1);
                const int lx = std::max(x - kernelSizeHalf, 0);
                const int rx = std::min(x + kernelSizeHalf, imgColor.width() - 1);
                //
                const auto setA = [](auto& A, const int si, const float ofsx, const float ofsy, const glm::vec3 albedo, const glm::vec3 normal) {
                    A(si, 0) = ofsx;
                    A(si, 1) = ofsy;
                    A(si, 2) = albedo.x;
                    A(si, 3) = albedo.y;
                    A(si, 4) = albedo.z;
                    A(si, 5) = normal.x;
                    A(si, 6) = normal.y;
                    A(si, 7) = normal.z;
                    A(si, 8) = 1.0f;
                };
                const auto setB = [](auto& b, const int si, const glm::vec3 color) {
                    b(si, 0) = color.x;
                    b(si, 1) = color.y;
                    b(si, 2) = color.z;
                };

                // ナイーブな最小二乗を解く
                int si = 0;
                for (auto yy : range(uy, dy)) {
                    for (auto xx : range(lx, rx)) {
                        const auto albedo = imgAlbedo(xx, yy);
                        const auto normal = imgNormal(xx, yy);
                        const auto color = imgColor(xx, yy);
                        setA(A, si, xx - x, yy - y, albedo, normal);
                        setB(b, si, color);
                        ++si;
                    }
                }
                Eigen::MatrixXf X = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

                if (false) {
                    // Tukeyロス関数でウェイトを計算し、ウェイト付きで解く
                    Eigen::MatrixXf matW = Eigen::MatrixXf::Zero(kernelSizeSqr, kernelSizeSqr);
                    std::vector<float> weights;
                    int32_t cnt = 0;
                    for (auto yy : range(uy, dy)) {
                        for (auto xx : range(lx, rx)) {
                            Eigen::MatrixXf aa = Eigen::MatrixXf::Zero(1, 9);
                            setA(aa, 0, xx - x, yy - y, imgAlbedo(xx, yy), imgNormal(xx, yy));
                            Eigen::MatrixXf bb = aa * X;
                            const float actual = toY(imgColor(xx, yy));
                            const float predict = toY(glm::vec3(bb(0), bb(1), bb(2)));
                            const float dist = std::abs(1.0f - (std::abs(predict - actual) / actual));
                            const float W = 1.0f;
                            const float tmp = std::max(1.0f - (dist * dist) / (W * W), 0.0f);
                            const float weight = tmp * tmp;
                            matW(cnt, cnt) = weight;
                            ++cnt;
                            weights.push_back(weight);
                        }
                    }
                    X = (matW * A).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(matW * b);
                }

                // 最終的な解を出す
                Eigen::MatrixXf aa = Eigen::MatrixXf::Zero(1, 9);
                setA(aa, 0, 0.0f, 0.0f, imgAlbedo(x, y), imgNormal(x, y));
                Eigen::MatrixXf bb = aa * X;
                dstImg(x, y) = glm::vec3(bb(0), bb(1), bb(2));
            }
        }

        // 進捗表示
        {
            const auto cur = lineCount.fetch_add(1);
            const auto progress0 = ((cur - 1) * 30) / imgColor.height();
            const auto progress1 = ((cur - 0) * 30) / imgColor.height();
            if (progress0 != progress1) {
                printf("=", progress1);
            }
        }
    });
    printf("] done\n");
    dstImg.save(denoisedFileName);
}
