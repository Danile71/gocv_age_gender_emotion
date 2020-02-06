package main

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

var (
	deviceID = "0"

	faceModel  = "data/face/opencv_face_detector_uint8.pb"
	faceConfig = "data/face/opencv_face_detector.pbtxt"

	emotionModel  = "data/emotion/EmotiW_VGG_S.caffemodel"
	emotionConfig = "data/emotion/deploy.prototxt"
	Emotions      = []string{"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"}

	ageModel  = "data/age/age_net.caffemodel"
	ageConfig = "data/age/age_deploy.prototxt"
	Ages      = []string{"0-2", "3-7", "8-12", "13-20", "20-36", "37-47", "48-55", "56-100"}

	genderModel  = "data/gender/gender_net.caffemodel"
	genderConfig = "data/gender/gender_deploy.prototxt"
	Genders      = []string{"Male", "Female"}
)

func main() {
	// open capture device
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	window := gocv.NewWindow("DNN Detection")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	// open DNN object tracking model
	faceNet := gocv.ReadNet(faceModel, faceConfig)
	if faceNet.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", faceModel, faceConfig)
		return
	}
	defer faceNet.Close()
	faceNet.SetPreferableBackend(gocv.NetBackendCUDA)
	faceNet.SetPreferableTarget(gocv.NetTargetCUDA)

	// open DNN object tracking model
	emotionNet := gocv.ReadNet(emotionModel, emotionConfig)
	if emotionNet.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", emotionModel, emotionConfig)
		return
	}
	defer emotionNet.Close()
	emotionNet.SetPreferableBackend(gocv.NetBackendCUDA)
	emotionNet.SetPreferableTarget(gocv.NetTargetCUDA)

	// open DNN object tracking model
	ageNet := gocv.ReadNet(ageModel, ageConfig)
	if emotionNet.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", emotionModel, emotionConfig)
		return
	}
	defer ageNet.Close()
	ageNet.SetPreferableBackend(gocv.NetBackendCUDA)
	ageNet.SetPreferableTarget(gocv.NetTargetCUDA)

	// open DNN object tracking model
	genderNet := gocv.ReadNet(genderModel, genderConfig)
	if genderNet.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", emotionModel, emotionConfig)
		return
	}
	defer genderNet.Close()
	genderNet.SetPreferableBackend(gocv.NetBackendCUDA)
	genderNet.SetPreferableTarget(gocv.NetTargetCUDA)

	var (
		ratio   float64 = 1.0
		mean            = gocv.NewScalar(104, 177, 123, 0)
		scalar          = gocv.NewScalar(0, 0, 0, 0)
		swapRGB         = false
	)
	fmt.Printf("Start reading device: %v\n", deviceID)

	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// convert image Mat to 300x300 blob that the object detector can analyze
		blob := gocv.BlobFromImage(img, ratio, image.Pt(300, 300), mean, swapRGB, false)

		// feed the blob into the detector
		faceNet.SetInput(blob, "data")

		// run a forward pass thru the network
		outputFace := faceNet.Forward("detection_out")

		for i := 0; i < outputFace.Total(); i += 7 {
			confidence := outputFace.GetFloatAt(0, i+2)

			if confidence > 0.5 {
				left := int(outputFace.GetFloatAt(0, i+3) * float32(img.Cols()))
				top := int(outputFace.GetFloatAt(0, i+4) * float32(img.Rows()))
				right := int(outputFace.GetFloatAt(0, i+5) * float32(img.Cols()))
				bottom := int(outputFace.GetFloatAt(0, i+6) * float32(img.Rows()))
				r := image.Rect(left, top, right, bottom)

				if r.Max.X < img.Cols() && r.Max.Y < img.Rows() && r.Min.X > 0 && r.Min.Y > 0 {
					gocv.Rectangle(&img, r, color.RGBA{0, 255, 0, 0}, 2)
					mat := img.Region(r)
					blob := gocv.BlobFromImage(mat, ratio, image.Pt(227, 227), scalar, swapRGB, false)
					//feed the blob into the detector
					emotionNet.SetInput(blob, "")
					// run a forward pass thru the network
					emoPreds := emotionNet.Forward("")
					_, _, _, emoLoc := gocv.MinMaxLoc(emoPreds)

					//feed the blob into the detector
					ageNet.SetInput(blob, "")
					// run a forward pass thru the network
					agePreds := ageNet.Forward("")
					_, _, _, ageLoc := gocv.MinMaxLoc(agePreds)

					//feed the blob into the detector
					genderNet.SetInput(blob, "")
					// run a forward pass thru the network
					genderPreds := genderNet.Forward("")
					_, _, _, genderLoc := gocv.MinMaxLoc(genderPreds)

					texts := []string{Genders[genderLoc.X], Ages[ageLoc.X], Emotions[emoLoc.X]}

					for i, text := range texts {
						size := gocv.GetTextSize(text, gocv.FontItalic, 1.2, 2)
						pt := image.Pt(r.Max.X, r.Min.Y+((i+1)*size.Y))
						gocv.PutText(&img, text, pt, gocv.FontHersheyComplexSmall, 1.2, color.RGBA{0, 0, 255, 0}, 2)
					}

					agePreds.Close()
					genderPreds.Close()
					emoPreds.Close()
					blob.Close()
					mat.Close()
				}
			}
		}

		outputFace.Close()
		blob.Close()

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
