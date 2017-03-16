from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, Http404, HttpResponseServerError
#from django.views.decorators.csrf import ensure_csrf_cookie
import csv, os
import json
import codecs
from datetime import datetime, date, timedelta
from tensor_model.models import *
from tensor_model.kalmanfilter import *
from tensor_model.tensorflow_models import *

import numpy as np
from django.template.loader import render_to_string

# Create your views here.

detector_list = ['KIN20701001','KIN20701002','KIN20701003','KIN20701004','KIN20701005','KIN20701006',
                 'KIN20701007','KIN20701008','KIN20701009','KIN20701010','KIN20701011','KIN20701012',
                 'KIN20701013','KIN20701014','KIN20701015','KIN20701016','KIN20701017','KIN20701018',
                 'KIN20701019','KIN20701020','KIN20701021','KIN20701022','KIN20701023','KIN20701024',
                 'KIN20701025','KIN20701026','KIN20700001','KIN20700002','KIN20700003','KIN20700004',
                 'KIN20700005','KIN20700006','KIN20700007','KIN20700008','KIN20700009','KIN20700010',
                 'KIN20700011','KIN20700012','KIN20700013','KIN20700014']

RSSI_default = [-999]*40

def preprocessing(data, tg_dt):        # for learning process
    list_for_regression = []
    make_exception = []
    time_max = tg_dt

    for i in range(0, len(data)):
        if (i in make_exception) == True:
            continue
        list_for_single_data = [data[i]['deviceData']['identifier'], data[i]['detectorData']['serial']]
        list_for_single_data.append([data[i]['receivedDate'] / 1000, float(data[i]['rssi'])])
        for j in range(i + 1, len(data)):
            if data[i]['deviceData']['identifier'] == data[j]['deviceData']['identifier'] and \
                            data[i]['detectorData']['serial'] == data[j]['detectorData']['serial']:
                list_for_single_data.append([data[j]['receivedDate'] / 1000, float(data[j]['rssi'])])
                make_exception.append(j)
        list_for_regression.append(list_for_single_data)

    list_exert_on_tf = []

    for k in range(0, len(list_for_regression)):
        # print(list_for_regression[k])
        # coordinates = np.array(sorted(list_for_regression[k][2:]))
        coordinates = np.array(list_for_regression[k][2:])

        if len(coordinates) > 3:
            x_origin = coordinates[:, 0]
            y_origin = coordinates[:, 1]
            x = x_origin[-3:]
            y = y_origin[-3:]

        else:
            x = coordinates[:, 0]
            y = coordinates[:, 1]

        z = np.polyfit(x, y, len(x) - 1)
        f = np.poly1d(z)

        x_new = np.linspace(min(x) - 1, time_max, (time_max - (min(x) - 1)) / 0.5)
        numsteps = int((time_max - (min(x) - 1)) / 0.5)

        y_new = f(x_new)

        A = np.matrix([1])          # state transition matrix
        H = np.matrix([1])          # control matrix
        B = np.matrix([0])          # observation matrix
        Q = np.matrix([0.003])      # estimated error in process (so supposed to be fixed)
        R = np.matrix([0.05])        # estimated error in measurements (measurement term is larger than learning term)
        xhat = np.matrix([y[0] * (1 - 0.4 * (np.random.rand(1) - 0.5))])
        P = np.matrix([1])          # initial covariance estimate

        filter = KalmanFilterLinear(A, B, H, xhat, P, Q, R)
        rssimeter = RSSImeter(1.20, 0.20)

        measuredRSSI = []
        kalmanRSSI = []

        for l in range(max(0, numsteps - 50), numsteps):
            measured = y_new[l]
            measuredRSSI.append(measured)
            kalmanRSSI.append(filter.GetCurrentState()[0, 0])
            filter.Step(np.matrix([0]), np.matrix([measured]))
        time_str = datetime.fromtimestamp(tg_dt).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        list_exert_on_tf.append([list_for_regression[k][0], list_for_regression[k][1], kalmanRSSI[-1], time_str])

    pre_tf_data = []
    skip_list = []

    for m in range(0, len(list_exert_on_tf)):
        if m in skip_list:
            continue
        one_beacon_data = [list_exert_on_tf[m][0], list_exert_on_tf[m][3]]
        RSSI_list = RSSI_default

        for n in range(m, len(list_exert_on_tf)):
            if list_exert_on_tf[n][0] == one_beacon_data[0]:
                RSSI_list[detector_list.index(list_exert_on_tf[n][1])] = list_exert_on_tf[n][2]
                skip_list.append(n)
            else:
                continue
        RSSI = np.array(RSSI_list)
        one_beacon_data.append(RSSI)
        pre_tf_data.append(one_beacon_data)

    return pre_tf_data

def do_filter_data(data, tg_dt):        # for location measurement
    list_for_regression = []
    make_exception = []
    time_max = tg_dt

    for i in range(0, len(data)):
        if (i in make_exception) == True:
            continue
        list_for_single_data = [data[i]['deviceData']['identifier'], data[i]['detectorData']['serial']]
        list_for_single_data.append([data[i]['receivedDate'] / 1000, float(data[i]['rssi'])])
        for j in range(i + 1, len(data)):
            if data[i]['deviceData']['identifier'] == data[j]['deviceData']['identifier'] and \
                            data[i]['detectorData']['serial'] == data[j]['detectorData']['serial']:
                list_for_single_data.append([data[j]['receivedDate'] / 1000, float(data[j]['rssi'])])
                make_exception.append(j)
        list_for_regression.append(list_for_single_data)

    list_exert_on_tf = []

    for k in range(0, len(list_for_regression)):
        # print(list_for_regression[k])
        # coordinates = np.array(sorted(list_for_regression[k][2:]))
        coordinates = np.array(list_for_regression[k][2:])

        if len(coordinates) > 8:
            x_origin = coordinates[:, 0]
            y_origin = coordinates[:, 1]
            x = x_origin[-8:]
            y = y_origin[-8:]

        else:
            x = coordinates[:, 0]
            y = coordinates[:, 1]

        z = np.polyfit(x, y, len(x) - 1)
        f = np.poly1d(z)

        x_new = np.linspace(min(x) - 1, time_max, (time_max - (min(x) - 1)) / 0.5)
        numsteps = int((time_max - (min(x) - 1)) / 0.5)

        y_new = f(x_new)

        A = np.matrix([1])          # state transition matrix
        H = np.matrix([1])          # control matrix
        B = np.matrix([0])          # observation matrix
        Q = np.matrix([0.003])      # estimated error in process (so supposed to be fixed)
        R = np.matrix([0.2])        # estimated error in measurements (measurement term is larger than learning term)
        xhat = np.matrix([y[0] * (1 - 0.4 * (np.random.rand(1) - 0.5))])
        P = np.matrix([1])          # initial covariance estimate

        filter = KalmanFilterLinear(A, B, H, xhat, P, Q, R)
        rssimeter = RSSImeter(1.20, 0.20)

        measuredRSSI = []
        kalmanRSSI = []

        for l in range(max(0, numsteps - 50), numsteps):
            measured = y_new[l]
            measuredRSSI.append(measured)
            kalmanRSSI.append(filter.GetCurrentState()[0, 0])
            filter.Step(np.matrix([0]), np.matrix([measured]))
        time_str = datetime.fromtimestamp(tg_dt).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        list_exert_on_tf.append([list_for_regression[k][0], list_for_regression[k][1], kalmanRSSI[-1], time_str])

    pre_tf_data = []
    skip_list = []

    for m in range(0, len(list_exert_on_tf)):
        if m in skip_list:
            continue
        one_beacon_data = [list_exert_on_tf[m][0], list_exert_on_tf[m][3]]
        RSSI_list = RSSI_default

        for n in range(m, len(list_exert_on_tf)):
            if list_exert_on_tf[n][0] == one_beacon_data[0]:
                RSSI_list[detector_list.index(list_exert_on_tf[n][1])] = list_exert_on_tf[n][2]
                skip_list.append(n)
            else:
                continue
        RSSI = np.array(RSSI_list)
        one_beacon_data.append(RSSI)
        pre_tf_data.append(one_beacon_data)

    return pre_tf_data


def do_tensorflow_learning(data, pre_tf_data):
    test_RSSI = []

    for pre_data in pre_tf_data:
        test_RSSI.append(pre_data[2])

    result = MDS_train_load(test_RSSI)

    # print(result)

    tf_data = pre_tf_data

    for index in range(0, len(tf_data)):
        section_number = int(result[index])
        (x_position, y_position) = (
        section_number % 44 + 1, int(section_number / 44) + 1)  # supposition: x-position[1, 44], y-position[1, 30]
        tf_data[index][-1] = section_number
        tf_data[index].append(x_position)
        tf_data[index].append(y_position)
        tf_data[index].append(data[0]["detectorData"]["showroomId"])
        tf_data[index].append(data[0]["detectorData"]["showroomName"])

    # print(tf_data)

    subjects = ['identifier', 'predictionDate', 'sectionNumber', 'xPosition', 'yPosition', 'showroomId', 'showroomName']
    location_data = []
    for lists in tf_data:
        location_dict = dict((subjects[i], lists[i]) for i in range(0, len(lists)))
        location_data.append(location_dict)

    print(location_data)
    return location_data


def main_page(request):
    """
    r_val = {'thisis': 'mainpage'}

    r = HttpResponse(content_type='application/json')
    r.write(json.dumps(r_val, indent=4))
    return r
    """
    return render(request, 'tensor_model/main_page.html', {})


def insert_page(request):
    r_val = {}

    reader = codecs.getreader("utf-8")
    data = json.load(reader(request))

    print(data)

    if data:

        # 비콘 별로 필터링

        time_list = []
        for i in range(0, len(data)):
            time_list.append(data[i]['receivedDate'] / 1000)
        # tg_dt_str = max(time_list)
        # tg_dt = datetime.strptime(tg_dt_str, "%Y-%m-%d %H:%M:%S")
        tg_dt = max(time_list)
        print(tg_dt)

        filtered_data = do_filter_data(data, tg_dt)
        learned_data = do_tensorflow_learning(data, filtered_data)
        r_val['data'] = learned_data
        r_val['status'] = 0
        r_val['error'] = None

        # Tensor Flow 가동

    else:
        r_val['data'] = []
        r_val['status'] = 1
        r_val['error'] = 'received empty data'

    # 최종 리턴

    r = HttpResponse(content_type='application/json')
    r.write(json.dumps(r_val, indent=4))
    return r


def prelearning_page(request):
    r_val = {}

    print(request)
    reader = codecs.getreader("utf-8")
    data = json.load(reader(request))

    print(data)

    if data:

        # 비콘 별로 필터링

        time_list = []
        for i in range(len(data)):
            #print(data[i], type(data[i]))
            time_list.append(data[i]['receivedDate'] / 1000)
        tg_dt = max(time_list)

        filtered_data = preprocessing(data, tg_dt)

        listed_data = []
        for single_list in filtered_data:
            single_data = {}
            single_data['identifier'] = single_list[0]
            single_data['predictionDate'] = single_list[1]
            RSSI_float = list(single_list[2])
            RSSI = []

            for num in range(0, len(detector_list)):
                one_detector = {}
                one_detector['serial'] = detector_list[num]
                one_detector['rssi'] = RSSI_float[num]

                RSSI.append(one_detector)

            single_data['signal'] = RSSI
            listed_data.append(single_data)

        r_val['data'] = listed_data
        r_val['status'] = 0
        r_val['error'] = None

    else:
        r_val['data'] = []
        r_val['status'] = 1
        r_val['error'] = 'received empty data'

    # print(r_val)
    r = HttpResponse(content_type='application/json')
    r.write(json.dumps(r_val, indent=4))
    return r

# @ensure_csrf_cookie
def data_page(request):
    reader = codecs.getreader("utf-8")
    raw_data = json.load(reader(request))

    r = HttpResponse(content_type='application/json')
    r.write(json.dumps(raw_data, indent=4))
    return r
