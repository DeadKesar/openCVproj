using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;

namespace openCVproj
{
    public partial class MainWindow : Window
    {
        private VideoCapture _capture = null;
        private bool _isCapturing = false;
        private CascadeClassifier _faceCascade;
        private CascadeClassifier _eyeCascade;

        public MainWindow()
        {
            InitializeComponent();

            // Инициализация классификаторов Haar
            try
            {
                string faceCascadePath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "haarcascade_frontalface_default.xml");
                string eyeCascadePath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "haarcascade_eye.xml");

                _faceCascade = new CascadeClassifier(faceCascadePath);
                _eyeCascade = new CascadeClassifier(eyeCascadePath);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Ошибка при загрузке классификаторов: " + ex.Message);
            }
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_isCapturing)
            {
                try
                {
                    _capture = new VideoCapture(0); // 0 - индекс камеры
                    _isCapturing = true;
                    StartButton.IsEnabled = false;
                    StopButton.IsEnabled = true;

                    await Task.Run(() => ProcessFrames());
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Ошибка при запуске камеры: " + ex.Message);
                }
            }
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            if (_isCapturing)
            {
                _isCapturing = false;
                StartButton.IsEnabled = true;
                StopButton.IsEnabled = false;

                _capture?.Dispose();
                _capture = null;
            }
        }

        private void ProcessFrames()
        {
            while (_isCapturing)
            {
                try
                {
                    using (Mat frame = _capture.QueryFrame())
                    {
                        if (frame != null)
                        {
                            // Преобразование в серый масштаб
                            Mat grayFrame = new Mat();
                            CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
                            CvInvoke.EqualizeHist(grayFrame, grayFrame);

                            // Обнаружение лиц
                            var faces = _faceCascade.DetectMultiScale(
                                grayFrame,
                                1.1,
                                10,
                                System.Drawing.Size.Empty);

                            foreach (var face in faces)
                            {
                                // Рисуем прямоугольник вокруг лица
                                CvInvoke.Rectangle(frame, face, new MCvScalar(0, 0, 255), 2); // Красный цвет

                                // Область интереса для глаз внутри лица
                                var faceROI = new Mat(grayFrame, face);
                                var eyes = _eyeCascade.DetectMultiScale(
                                    faceROI,
                                    1.1,
                                    10,
                                    System.Drawing.Size.Empty);

                                foreach (var eye in eyes)
                                {
                                    // Корректируем координаты глаз относительно всего кадра
                                    Rectangle eyeRect = new Rectangle(
                                        face.X + eye.X,
                                        face.Y + eye.Y,
                                        eye.Width,
                                        eye.Height);

                                    // Рисуем прямоугольник вокруг глаза
                                    CvInvoke.Rectangle(frame, eyeRect, new MCvScalar(255, 0, 0), 2); // Синий цвет
                                }
                            }

                            // Обновление изображения в UI
                            Dispatcher.Invoke(() =>
                            {
                                VideoImage.Source = ToBitmapSource(frame);
                            });
                        }
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() =>
                    {
                        MessageBox.Show("Ошибка при обработке кадра: " + ex.Message);
                        StopButton_Click(null, null);
                    });
                }
            }
        }


        private BitmapSource ToBitmapSource(Mat mat)
        {
            using (Bitmap bitmap = mat.ToBitmap())
            {
                var bitmapData = bitmap.LockBits(
                    new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                    System.Drawing.Imaging.ImageLockMode.ReadOnly,
                    bitmap.PixelFormat);

                BitmapSource bitmapSource = BitmapSource.Create(
                    bitmap.Width,
                    bitmap.Height,
                    bitmap.HorizontalResolution,
                    bitmap.VerticalResolution,
                    System.Windows.Media.PixelFormats.Bgr24,
                    null,
                    bitmapData.Scan0,
                    bitmapData.Stride * bitmap.Height,
                    bitmapData.Stride);

                bitmap.UnlockBits(bitmapData);
                return bitmapSource;
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            _isCapturing = false;
            _capture?.Dispose();
        }
    }
}
