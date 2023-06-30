from flask import Flask,render_template,request
from src.pipeline.predictionpipeline import customdata,predictpipeline
application=Flask(__name__)
app=application
@app.route('/')
def homepage():
     return render_template('home.html')
@app.route('/submit',methods=['POST'])
def predictpage():
     if request.method=='POST':
          cd=customdata(
               carat=float(request.form.get('carat')),
               depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
          )
          df=cd.get_data_as_dataframe()
     obj=predictpipeline()
     result=obj.predict(df)
     predprice=round(float(result),2)
     return render_template('submit.html',predprice=predprice)


if __name__=='__main__':
     app.run(host='0.0.0.0',port=80)
