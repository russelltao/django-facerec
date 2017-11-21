from django.db import models
from django.core.files.storage import FileSystemStorage

fs = FileSystemStorage(location='/media/photos')

# Create your models here.
class UserFace(models.Model):
    """
    注册人脸
    """
    name = models.CharField(verbose_name='姓名', max_length=80, unique=True)
    facepic = models.ImageField(verbose_name='注册照片',upload_to='reg')
    embeddings = models.CharField(verbose_name='特征向量', max_length=4096, null=True)

    def image(self):
        return '<img src="/media/%s"/>' % self.facepic

    image.allow_tags = True

    class Meta:
        verbose_name = "注册人脸"

    def __str__(self):
        return self.name

class RecognizeRecord(models.Model):
    """
    识别人脸记录
    """
    pic = models.ImageField(verbose_name='待识别照片',upload_to='face', unique=True)
    similiar1 = models.FloatField(verbose_name='第一人相似度')
    user1 = models.ForeignKey(UserFace, verbose_name='第一相似人', null=True, related_name='user1')
    similiar2 = models.FloatField(verbose_name='第二人相似度')
    user2 = models.ForeignKey(UserFace, verbose_name='第二相似人', null=True, related_name='user2')
    result = models.IntegerField(default=0, verbose_name='识别结果')

    def image(self):
        return '<img src="/media/%s"/>' % self.pic

    image.allow_tags = True

    def image1(self):
        return '<img src="/media/%s"/>' % self.user1.facepic

    image1.allow_tags = True

    def image2(self):
        return '<img src="/media/%s"/>' % self.user2.facepic

    image2.allow_tags = True

    class Meta:
        verbose_name = "facenet识别记录"
