from flask import jsonify, render_template, redirect, request, url_for
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user
)

from app import db, login_manager
from app.base import blueprint
from app.base.forms import LoginForm, CreateAccountForm
from app.base.models import User, Industry, NetExports

from app.base.util import verify_pass

@blueprint.route('/')
def route_default():
    return redirect(url_for('base_blueprint.login'))

## Login & Registration

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:
        
        # read form data
        username = request.form['username']
        password = request.form['password']

        # Locate user
        user = User.query.filter_by(username=username).first()
        
        # Check the password
        if user and verify_pass( password, user.password):

            login_user(user)
            return redirect(url_for('base_blueprint.route_default'))

        # Something (user or pass) is not ok
        return render_template( 'accounts/login.html', msg='Wrong user or password', form=login_form)

    if not current_user.is_authenticated:
        return render_template( 'accounts/login.html',
                                form=login_form)
    return redirect(url_for('home_blueprint.index'))

@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    login_form = LoginForm(request.form)
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:

        username  = request.form['username']
        email     = request.form['email'   ]

        # Check usename exists
        user = User.query.filter_by(username=username).first()
        if user:
            return render_template( 'accounts/register.html', 
                                    msg='Username already registered',
                                    success=False,
                                    form=create_account_form)

        # Check email exists
        user = User.query.filter_by(email=email).first()
        if user:
            return render_template( 'accounts/register.html', 
                                    msg='Email already registered', 
                                    success=False,
                                    form=create_account_form)

        # else we can create the user
        user = User(**request.form)
        db.session.add(user)
        db.session.commit()

        return render_template( 'accounts/register.html', 
                                msg='User created please <a href="/login">login</a>', 
                                success=True,
                                form=create_account_form)

    else:
        return render_template( 'accounts/register.html', form=create_account_form)

@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('base_blueprint.login'))

@blueprint.route('/shutdown')
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

## Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('page-403.html'), 403

@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('page-403.html'), 403

@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('page-404.html'), 404

@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('page-500.html'), 500

@blueprint.route('/records', methods=['GET', 'POST'])
@login_required
def records():
    if request.method=="POST":
        year =  request.form['year']
        agric =  request.form['agric']
        mining =  request.form['mining']
        manufacturing =  request.form['manufacturing']
        electricity_water =  request.form['electricity_water']
        construction =  request.form['construction']
        distribution =  request.form['distribution']
        transport =  request.form['transport']
        financial =  request.form['financial']
        real_estate =  request.form['real_estate']
        public_administration =  request.form['public_administration']
        education =  request.form['education']
        human_health =  request.form['human_health']
        domestic_services =  request.form['domestic_services']
        net_tax =  request.form['net_tax']
        GDP =  request.form['GDP']
        industry = Industry(year=year,agric=agric,mining=mining,manufacturing=manufacturing,electricity_water=electricity_water,
                          construction=construction,distribution=distribution,transport=transport,financial=financial,
                          real_estate=real_estate,public_administration=public_administration,education=education,
                          human_health=human_health,domestic_services=domestic_services,net_tax=net_tax,GDP=GDP)
        db.session.add(industry)
        db.session.commit()
        return redirect('/records') 
    industrial_factors = Industry.query.all()
    print(industrial_factors)
    return render_template( 'db_records/records.html', industrial_factors=industrial_factors)

@blueprint.route('/records_bop', methods=['GET', 'POST'])
@login_required
def records_bop():
    if request.method=="POST":
        year =  request.form['year']
        goods_imports =  request.form['goods_imports']
        goods_exports =  request.form['goods_exports']
        services_exports =  request.form['services_exports']
        services_imports =  request.form['services_imports']
        bop_goods =  request.form['bop_goods']
        bop_services =  request.form['bop_services']
        bop = NetExports(year=year,goods_imports=goods_imports,goods_exports=goods_exports,services_exports=services_exports,
                        services_imports=services_imports,bop_goods=bop_goods,bop_services=bop_services)
        db.session.add(bop)
        db.session.commit()
        return redirect('/records_bop')
    bops = NetExports.query.all()
    print(bops)
    return render_template( 'db_records/bop.html', bops=bops)

