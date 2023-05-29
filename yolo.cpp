// title : yolo.cpp
// date : 2021.11.17. created, 2023.5.19 updated
// author : sungryul lee

#include "yolo.hpp"

static std::vector<String> output_names;
static std::vector<std::string> class_names;
//static cv::dnn::Net net;

static const float CONFIDENCE_THRESHOLD = 0.3;
static const float NMS_THRESHOLD = 0.2;
static const int NUM_CLASSES = 80;

// colors for bounding boxes
static const cv::Scalar colors[] = {
	{0, 255, 255},
	{255, 255, 0},
	{0, 255, 0},
	{255, 0, 0}
};
static const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

Net yolo_init(const String& cfgfile, const String& weightfile, const String& classnames)
{
	//std::ifstream class_file("robot.names");
	std::ifstream class_file(classnames);
	if (!class_file){ std::cerr << "failed to open classes.txt\n"; exit(1); }
	std::string line;
	while (std::getline(class_file, line)) class_names.push_back(line);
	
	//auto net = cv::dnn::readNetFromDarknet("yolov4-tiny-robot.cfg", "yolov4-tiny-robot_2000.weights");
	//net = cv::dnn::readNetFromDarknet("yolov4-tiny-robot-288.cfg", "yolov4-tiny-robot-288_final.weights");
	//auto net = cv::dnn::readNetFromDarknet("yolov4-robot.cfg", "yolov4-robot_final.weights");
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfgfile, weightfile);
	if(net.empty()) { cerr << "Network load failed" <<endl; exit(1);}

	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	//net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	//net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	//auto output_names = net.getUnconnectedOutLayersNames();
	output_names = net.getUnconnectedOutLayersNames();
	return net;
}

int yolo_inference(Net& net,cv::Size size, Mat& frame)
{
	cv::Mat blob;
	std::vector<cv::Mat> detections;
	
	//cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
	//cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(288, 288), cv::Scalar(), true, false, CV_32F);
	cv::dnn::blobFromImage(frame, blob, 0.00392, size, cv::Scalar(), true, false, CV_32F);
	net.setInput(blob);

	net.forward(detections,output_names);

	std::vector<int> indices[NUM_CLASSES];
	std::vector<cv::Rect> boxes[NUM_CLASSES];
	std::vector<float> scores[NUM_CLASSES];

	for (auto& output : detections)
	{
		const auto num_boxes = output.rows;
		for (int i = 0; i < num_boxes; i++)
		{
			auto x = output.at<float>(i, 0) * frame.cols;
			auto y = output.at<float>(i, 1) * frame.rows;
			auto width = output.at<float>(i, 2) * frame.cols;
			auto height = output.at<float>(i, 3) * frame.rows;
			cv::Rect rect(x - width / 2, y - height / 2, width, height);
			for (int c = 0; c < NUM_CLASSES; c++)
			{
				auto confidence = *output.ptr<float>(i, 5 + c);
				if (confidence >= CONFIDENCE_THRESHOLD)
				{
					boxes[c].push_back(rect);
					scores[c].push_back(confidence);
					//cout<<"detect"<<endl;
				}
			}
		}
	}

	for (int c = 0; c < NUM_CLASSES; c++)
	{
		//cout << boxes[c].size() <<','<< scores[c].size()<<endl;
		cv::dnn::NMSBoxes(boxes[c], scores[c], 0.2, NMS_THRESHOLD, indices[c]);
	}
	for (int c = 0; c < NUM_CLASSES; c++)
	{
		//cout << indices[c].size() << endl;
		for (size_t i = 0; i < indices[c].size(); ++i)
		{
			
			const auto color = colors[c % NUM_COLORS];

			auto idx = indices[c][i];
			const auto& rect = boxes[c][idx];
			cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 2);
			
			std::ostringstream label_ss;
			label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
			auto label = label_ss.str();

			int baseline;
			auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
			cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
			cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
		}
	}
	return detections.size();
}










