import boto3


class FaceDetection:
    """
    This class will recognize face using aws service
    """

    def __init__(self, file_path, access_key, access_secret, access_token):
        """

        :param file_path:
        """
        self.photo = file_path
        self.aws_access_key_id = access_key
        self.aws_access_secret = access_secret
        self.access_token = access_token

        # initialize boto3 object
        self.client = boto3.client("rekognition", aws_access_key_id=self.aws_access_key_id,
                                   aws_secret_access_key=self.aws_access_secret,
                                   aws_session_token=self.access_token
                                   )

    def face_detection(self):
        """

        :return:
        """
        # read the image file passed
        with open(self.photo, 'rb') as img:
            response = self.client.detect_faces(Image={
                'Bytes': img.read()
            },
                Attributes=['ALL']
            )

        # parse the result to get the required fields
        if response['FaceDetails']:
            result = {
                'gender': response['FaceDetails'][0]['Gender']['Value'],
                'age_range': str(response['FaceDetails'][0]['AgeRange']['Low']) + '-' + str(
                    response['FaceDetails'][0]['AgeRange']['High']),
                'emotions': response['FaceDetails'][0]['Emotions'][0]['Type'],
                'sun_glasses': response['FaceDetails'][0]['Sunglasses']['Value'],
                'beard': response['FaceDetails'][0]['Beard']['Value'],
                'mustache': response['FaceDetails'][0]['Mustache']['Value'],
                'eyes_open': response['FaceDetails'][0]['EyesOpen']['Value'],
                'mouth_open': response['FaceDetails'][0]['MouthOpen']['Value'],
                'smile': response['FaceDetails'][0]['Smile']['Value']
            }
        else:
            result = 'Face could not be detected'
        return result

    def run(self):
        details = self.face_detection()

        # display the output
        if isinstance(details, dict):
            print('**************************The candidate has the following details*********************************')
            for each_key in details.keys():
                print(each_key + '=' + str(details[each_key]))

        else:
            print(details)


if __name__ == '__main__':
    # Initializing class and creating a class object
    fd = FaceDetection(file_path='/home/spandey/Downloads/image.png',
                       access_key='ASIAYCBWE4BR2ICTPWFR',
                       access_secret='sULpD3NX66EpX/TBY8ZbtgI0mVIkGxq8f80PgcLj',
                       access_token='FwoGZXIvYXdzEFsaDCEYgEpuOVCxFHg65CLXAfMmf31ovHR5440ICGYU7swXY2ch/sJ41KVn5bvdRevnmDDWhh1ebZRB/DAsdGyMZ/Mu5tE1npyrJ0c8Gtqfmwd3raV0r0VwORSpSpNozMvAyztKpB8Ls2W4hmid+grIVrACAxdhYmTZCTStdt7vuuJKc+QBDp5YI85PFChSKzNzkxF12QZJ095YN+P1QxwZgvBG9IjolBUthzO5CVFlUynsMF6fV2mebB4U32/IYGBNt1R1h0O9EftyVoZT5PkVHPgB+JG8hfVltNfZRmWJdDJlrPbMOVFgKI261vsFMi1jtgb/RBlAES9DH2hWK6upFnOtgF3os7Iy1ZHHbbrlXajo106B6HNqYxRUHqA=')
    fd.run()
