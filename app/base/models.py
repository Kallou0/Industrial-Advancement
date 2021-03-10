from flask_login import UserMixin
from sqlalchemy import Binary, Column, Integer, String

from app import db, login_manager

from app.base.util import hash_pass

class User(db.Model, UserMixin):

    __tablename__ = 'User'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password = Column(Binary)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]

            if property == 'password':
                value = hash_pass( value ) # we need bytes here (not plain str)
                
            setattr(self, property, value)

    def __repr__(self):
        return str(self.username)


@login_manager.user_loader
def user_loader(id):
    return User.query.filter_by(id=id).first()

@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    return user if user else None


class Industry(db.Model):
    __tablename__ = 'Industry'
    id = Column(Integer, primary_key=True)
    year =  Column(String(150), unique = True, nullable=False)
    agric =  Column(String(564), default='1000 ')
    mining =  Column(String(564), default='1000 ')
    manufacturing =  Column(String(564), default='1000 ')
    electricity_water =  Column(String(564), default='1000 ')
    construction =  Column(String(564), default='1000 ')
    distribution =  Column(String(564), default='1000 ')
    transport =  Column(String(564), default='1000 ')
    financial =  Column(String(564), default='1000 ')
    real_estate =  Column(String(564), default='1000 ')
    public_administration =  Column(String(564), default='1000 ')
    education =  Column(String(564), default='1000 ')
    human_health =  Column(String(564), default='1000 ')
    domestic_services =  Column(String(564), default='1000')
    net_tax =  Column(String(564), default='1000')
    GDP =  Column(String(564), default='1000 ')
    

    def __repr__(self):
        return self.GDP

class NetExports(db.Model):
    __tablename__ = 'NetExports'
    id = Column(Integer, primary_key=True)
    year =  Column(String(150), unique = True, nullable=False)
    goods_imports =  Column(String(564), default='1000 ')
    goods_exports =  Column(String(564), default='1000 ')
    services_exports =  Column(String(564), default='1000 ')
    services_imports =  Column(String(564), default='1000 ')
    bop_goods =  Column(String(564), default='1000 ')
    bop_services =  Column(String(564), default='1000 ')
    
    
    def __repr__(self):
        return self.bop_goods
