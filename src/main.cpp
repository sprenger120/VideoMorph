#include <iostream>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <string_view>
#include <filesystem>
#include <vector>
#include <ranges>
#include <tuple>
#include <chrono>

constexpr unsigned emoji_size_square_px = 16;

float matToBrightness(const cv::Mat &mat) {
    static cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_RGB2GRAY);
    const double sum_brightness = cv::sum(gray)[0];
    // Calculate the maximum possible sum (number of pixels * max pixel value, i.e., 255)
    const double max_brightness = gray.rows * gray.cols * 255.0;

    // Normalize to range [0, 1]
    return static_cast<float>(sum_brightness / max_brightness);
}

int main() {
    // open video
    static constexpr std::string_view currentFile = "1734952661415253.webm";
    cv::VideoCapture cap(currentFile.data(), cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        // check if we succeeded
        std::cout << "Unable to open file " << currentFile << "\n";
        return -1;
    }


    /// emojis
    // tuple<Image Content, LightValue>
    // LightValue: Dark = 0, Bright = 1
    std::vector<std::tuple<cv::Mat, float> > emojis;

    // load
    // down below every texture is compared pixel by pixel with the ROI of the vidoe frame
    // obv. this is very expensive
    // new method: write a scoring function that reduces the picture to average color (in HSV)
    // and the sigma value of how the colors are distributed
    static std::filesystem::path emoji_dir = std::filesystem::current_path() / "minecraft";
    for (const auto &file: std::filesystem::directory_iterator(emoji_dir)) {
        std::cout << "loaded: " << file.path().string() << "\n";
        auto emoji = cv::imread(file.path().string());
        cv::resize(emoji, emoji, cv::Size(emoji_size_square_px, emoji_size_square_px));
        emojis.emplace_back(emoji, matToBrightness(emoji));
    }
    // sort by brightness
    // lowest brightness at index 0
    std::ranges::sort(emojis, [](const auto &a, const auto &b) {
        return std::get<1>(a) < std::get<1>(b);
    });


    // grab total number of frames
    const int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total number of frames: " << totalFrames << std::endl;

    const int video_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));;
    const int video_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));;


    cv::VideoWriter writer("out2.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 24, cv::Size(video_w, video_h), true);


    // extract frame by frame
    cv::Mat frame(video_h, video_w, CV_8UC3);
    int frame_count = 0;

    cv::Mat emojified_frame(frame.size(), frame.type());

    // prepare diffs array
    std::vector<float> diffs;
    diffs.resize(emojis.size());
    cv::Mat diff_mat(emoji_size_square_px, emoji_size_square_px, frame.type());
    cv::Mat chunk(emoji_size_square_px, emoji_size_square_px, frame.type());

    while (true) {
        if (!cap.read(frame)) {
            std::cout << "End of video\n";
            break;
        }
        // blacken output frame to avoid graphics artiface
        emojified_frame = cv::Scalar(0, 0, 0);

        for (int col = 0; col < frame.cols - emoji_size_square_px; col += emoji_size_square_px) {
            for (int row = 0; row < frame.rows - emoji_size_square_px; row += emoji_size_square_px) {
                auto roi = cv::Rect(col, row, emoji_size_square_px, emoji_size_square_px);
                frame(roi).copyTo(chunk);

                for (auto elem : std::views::zip(diffs, emojis)) {
                    auto & [diff, emoji] = elem;
                    cv::absdiff(chunk, std::get<0>(emoji), diff_mat);
                    diff = matToBrightness(diff_mat);
                }
                auto res = std::min_element(diffs.begin(), diffs.end());
                auto emoji_index = std::distance(diffs.begin(), res);

                auto &selected_emoji = std::get<0>(emojis[emoji_index]);
                selected_emoji.copyTo(emojified_frame(roi));
            }
        }


        //cv::imshow("Emojified", emojified_frame);
        //cv::imshow("Frame", frame);
        //cv::waitKey(5);
        std::cout << (++frame_count) << "\n";

        writer.write(emojified_frame);
    }


    return 0;
}
