#!/usr/bin/env python

"""
Tool to help a human being to classify the scraped VPM pages 
 into several categories

It creates a GUI browser that shows sequentially one of the detected pages. 
 You can interact with the browser with the mouse and you can also use the 
 numerical pad of the keyboard to select one of the categories. Once you 
 have chosen the right category for a page the software moves to the next.

Author: Carlo Bottai
Copyright (c) 2020 - TU/e and EPFL
License: See the LICENSE file.
Date: 2020-10-16

"""

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
import qtawesome as qta
import sys
import webbrowser
from flata import Flata, Query, JSONStorage
import requests
from iris_utils.parse_args import parse_io


USER_AGENT = ('Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) '
              'Gecko/2009021910 Firefox/3.0.7')


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        args = parse_io()
        self.f_in = args.input
        
        self.read_data()

        self.view = QWebEngineView()
        self.view.settings() \
            .setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.setCentralWidget(self.view)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        
        navtb = QToolBar('Navigation')
        self.addToolBar(navtb)
        
        back_btn = QAction(qta.icon('fa5s.arrow-left'), 'Back', self)
        back_btn.triggered.connect(lambda: self.view.back())
        navtb.addAction(back_btn)

        next_btn = QAction(qta.icon('fa5s.arrow-right'), 'Forward', self)
        next_btn.triggered.connect(lambda: self.view.forward())
        navtb.addAction(next_btn)
       
        navtb.addSeparator()

        self.urlbar = QLineEdit()
        self.urlbar.returnPressed.connect(self.go_to_url)
        navtb.addWidget(self.urlbar)
       
        navtb.addSeparator()
       
        reload_btn = QAction(qta.icon('fa5s.redo'), 'Reload', self)
        reload_btn.triggered.connect(lambda: self.view.reload())
        navtb.addAction(reload_btn)

        stop_btn = QAction(qta.icon('fa5s.stop'), 'Stop', self)
        stop_btn.triggered.connect(lambda: self.view.stop())
        navtb.addAction(stop_btn)

        open_btn = QAction(
            qta.icon('fa5s.external-link-square-alt'), 'Open', self)
        open_btn.triggered.connect(lambda: \
            webbrowser.open_new_tab(self.urlbar.text()))
        navtb.addAction(open_btn)
        
        labtb = QToolBar('Labeling')
        self.addToolBar(Qt.RightToolBarArea, labtb)
        
        for name, idx in [
                ('VPM page | True patent-product link', 1),
                ('Brochure or description of the product | True patent-product link', 2),
                ('Hybrid document | True patent-product link', 3),
                ('List of patents or metadata of a patent | False patent-product link', 4),
                ('A scientific publication | False patent-product link', 5),
                ('News about the patent | False patent-product link', 6),
                ('CV/resume | False patent-product link', 7),
                ('Something else in a website to keep | False patent-product link', 8),
                ('Something else in a website to exclude | False patent-product link', 9),
                ('The document is unreachable | False patent-product link', 0)]:
            label = QAction(f'{name} ({idx})', self)
            label.setShortcut(str(idx))
            label.triggered.connect(lambda checked, lbl=name: self.label_page(lbl))
            labtb.addAction(label)
        
        #labtb.addSeparator()
        
        urls_len_lbl = f'{self.data_to_classify_len} URLs left to classify'
        self.status.showMessage(urls_len_lbl)

        self.open_next_page()

        self.show()

        self.setWindowTitle('VPM pages handmade classifier')

    def read_data(self):
        DB = Flata(self.f_in, storage=JSONStorage)
        self.database = DB.table('iris_vpm_pages_classifier')
        
        to_classify = \
            (Query().vpm_page_classification==None) & \
            (Query().vpm_page!=None)
        self.data_to_classify = iter(self.database.search(to_classify))
        self.data_to_classify_len = self.database.count(to_classify)

    def go_to_url(self, url=None):
        if url is None:
            url = self.urlbar.text()
        else:
            self.urlbar.setText(url)
            self.urlbar.setCursorPosition(0)
        
        try:
            response = requests.head(
                url, 
                headers={'User-Agent': USER_AGENT}, 
                verify=False, 
                allow_redirects=True,
                timeout=10)
            headers = response.headers
            content_type = headers['Content-Type']
            if 'Content-Disposition' in headers:
                content_disposition = headers['Content-Disposition']
            else:
                content_disposition = ''
            if not (content_type.startswith('text/html') or \
                    content_type.startswith('application/pdf') or \
                    content_type.startswith('text/plain')) or \
               content_disposition.startswith('attachment'):
                self.msgBox = QMessageBox.about(
                    self, 
                    'Additional information (DOWNLOAD)', 
                    ('It is possible that it is needed to download the next '
                     'document.\nIf you do not see the page changing, try to '
                     'open the page in a browser by clicking on '
                     'the appropriate button'))
        except:
            pass
        
        url = QUrl(url)
        
        if url.scheme() == '':
            url.setScheme('https')

        self.view.setUrl(url)

    def open_next_page(self):
        try:
            self.current_data = next(self.data_to_classify)
            while self.current_data['vpm_page_classification']:
                self.current_data = next(self.data_to_classify)
            
            INFO_MSG = {
                'COPYRIGHT': 
                    ('The information about the patent(s) has been '
                     'detected close to the copyright information '
                     'at the bottom of the document.\n'
                     'Please, confirm whether or not there is a link '
                     'between a patent and a product in this document'),
                'NOCORPUS': 
                    ('No information about any of the patents has been '
                     'detected in the document.\nPlease, confirm whether '
                     'or not there is a link between a patent and a '
                     'product in this document'),
                'NOCORPUS+IMG': 
                    ('The only information about the patent(s) '
                     'has been detected in one of the pictures '
                     'of the document.\nPlease, confirm whether '
                     'or not there is a link between a patent '
                     'and a product in this document'),
                'NOCORPUS+PATNUMINURL': 
                    ('The only information about the patent(s) '
                     'has been detected in the URL '
                     'of the document.\nPlease, confirm whether '
                     'or not there is a link between a patent '
                     'and a product in this document')}
            vpm_page_automatic_classification = self.current_data[
                'vpm_page_automatic_classification']
            vpm_page_automatic_classification_info = \
                vpm_page_automatic_classification \
                    .split(' | ')[1]
            if vpm_page_automatic_classification_info in INFO_MSG.keys():
                vpm_page_automatic_classification_msg = INFO_MSG[
                    vpm_page_automatic_classification_info]
                self.msgBox = QMessageBox.about(
                    self, 
                    f'Additional information ({vpm_page_automatic_classification_info})', 
                    vpm_page_automatic_classification_msg)
            
            print('\n+++++++++++++++++++++++++++')
            print(f"Patent assignee: {self.current_data['patent_assignee']}")
            try:
                print(f"Award recipient: {self.current_data['award_recipient']}")
            except Exception:
                pass
            print(f"Patents: {self.current_data['patent_id']}")
            print('+++++++++++++++++++++++++++\n')
            
            url = self.current_data['vpm_page']
            self.go_to_url(url)

        except:
            print('\n+++++++++++++++++++++++++++')
            print('No other pages left. Well done!')
            print('+++++++++++++++++++++++++++\n')
            self.close()
    
    def label_page(self, label):
        updated_info = self.database.update(
            {'vpm_page_classification': label}, 
            Query().vpm_page==self.current_data['vpm_page'])
        updated_ids = updated_info[0]
        
        # Reduce the number of pages left by one 
        #   and show this information in the status bar
        self.data_to_classify_len -= len(updated_ids)
        urls_len_lbl = f'{self.data_to_classify_len} URLs left to classify'
        self.status.showMessage(urls_len_lbl)

        self.open_next_page()
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName('VPM pages handmade classifier')
    window = MainWindow()
    app.exec_()
    
