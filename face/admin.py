from django.contrib import admin
from .models import *
from django.db import models
from form_utils.widgets import ImageWidget

@admin.register(UserFace)
class UserFaceAdmin(admin.ModelAdmin):
    list_display = ['name','image','embeddings']
    formfield_overrides = {models.ImageField: {'widget': ImageWidget}}
    search_fields = ['name', ]

@admin.register(RecognizeRecord)
class RecognizeRecordAdmin(admin.ModelAdmin):
    list_display = ['image','similiar1','image1','user1','similiar2','image2','user2','result']
    formfield_overrides = {models.ImageField: {'widget': ImageWidget}}
    list_filter = ['result']
