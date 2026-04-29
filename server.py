from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os, logging, uuid, bcrypt, jwt, json as jsonlib
from pathlib import Path
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from emergentintegrations.llm.chat import LlmChat, UserMessage
from emergentintegrations.payments.stripe.checkout import StripeCheckout, CheckoutSessionRequest

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'decant_department')]
SECRET_KEY = os.environ.get('JWT_SECRET', 'decant-department-secret-key-2025')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')
STRIPE_KEY = os.environ.get('STRIPE_API_KEY', '')
LABEL_FEE = 5.00
ADMIN_EMAIL = "admin@decantdepartment.com"
ADMIN_PASSWORD_RAW = "DecantAdmin#2025"
ADMIN_NAME = "Cologne Market Admin"
EARLY_ACCESS_FEE = 0.01  # 1% - seller keeps 99%
NORMAL_FEE = 0.10  # 10% - seller keeps 90%
SHIPPING_FEE_BUYER = 5.00
LOGO_URL = "https://customer-assets.emergentagent.com/job_scent-market-64/artifacts/8uicmibv_ChatGPT%20Image%20Apr%2026%2C%202026%2C%2012_43_11%20PM.png"

def clean_doc(doc):
    if doc and "_id" in doc: del doc["_id"]
    return doc
def clean_docs(docs): return [clean_doc(d) for d in docs]

app = FastAPI(title="Cologne Market API")
api_router = APIRouter(prefix="/api")
security = HTTPBearer()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== MODELS ==============
class UserCreate(BaseModel):
    email: EmailStr; password: str; name: str
class UserLogin(BaseModel):
    email: EmailStr; password: str
class UserResponse(BaseModel):
    id: str; email: str; name: str; is_admin: bool = False
    balance: float = 0.0; created_at: datetime
class TokenResponse(BaseModel):
    access_token: str; token_type: str = "bearer"; user: UserResponse
class ListingCreate(BaseModel):
    brand: str; name: str; concentration: str
    full_bottle_size_ml: int; remaining_ml: int
    notes: str; image: Optional[str] = None; batch_year: Optional[int] = None; price_per_ml: float
class ListingUpdate(BaseModel):
    notes: Optional[str] = None; price_per_ml: Optional[float] = None
    remaining_ml: Optional[int] = None; image: Optional[str] = None; status: Optional[str] = None
class ListingResponse(BaseModel):
    id: str; seller_id: str; seller_name: str; brand: str; name: str; concentration: str
    full_bottle_size_ml: int; remaining_ml: int; available_ml: int; fill_percent: float
    notes: str; image: Optional[str] = None; batch_year: Optional[int] = None; status: str
    price_per_ml: float; platform_fee_percent: float = 0.10; fee_type: str = "normal"
    authenticity_status: str = "pending"; authenticity_note: Optional[str] = None; created_at: datetime
class OrderResponseV2(BaseModel):
    id: str; listing_id: str; listing_brand: str; listing_name: str
    listing_concentration: str; listing_image: Optional[str] = None
    seller_id: str; seller_name: str; buyer_name: str; buyer_email: str
    selected_size_ml: int; price_paid: float; seller_earnings: float; status: str; created_at: datetime
class CheckoutRequest(BaseModel):
    listing_id: str; selected_size_ml: int; buyer_name: str; buyer_email: EmailStr; origin_url: str
class PaymentTransactionResponse(BaseModel):
    id: str; order_id: str; stripe_session_id: str; listing_id: str
    listing_brand: str; listing_name: str; seller_id: str; seller_name: str
    buyer_name: str; buyer_email: str; selected_size_ml: int
    product_price: float; platform_fee: float; shipping_fee: float; total_amount: float
    seller_net_earnings: float; payment_status: str; payout_status: str
    stripe_transfer_id: Optional[str] = None; created_at: datetime
class ConversationCreate(BaseModel):
    recipient_id: str; listing_id: Optional[str] = None
class MessageCreate(BaseModel):
    text: str
class MessageResponse(BaseModel):
    id: str; conversation_id: str; sender_id: str; sender_name: str; text: str; read: bool; created_at: datetime
class ConversationResponse(BaseModel):
    id: str; participants: List[str]; participant_names: dict
    listing_id: Optional[str] = None; listing_name: Optional[str] = None; listing_brand: Optional[str] = None
    last_message: str; last_message_at: Optional[datetime] = None; unread_count: int = 0; created_at: datetime
class LabelRequestCreate(BaseModel):
    order_id: str; seller_address: dict
class LabelRequestResponse(BaseModel):
    id: str; order_id: str; seller_id: str; seller_name: str; seller_address: dict
    listing_brand: str; listing_name: str; buyer_name: str; status: str
    label_url: Optional[str] = None; tracking_number: Optional[str] = None; carrier: Optional[str] = None
    label_fee_status: str = "pending"; created_at: datetime
class ReturnRequestCreate(BaseModel):
    listing_id: str; reason: Optional[str] = None
class ReturnResponse(BaseModel):
    id: str; seller_id: str; listing_id: str; listing_brand: str; listing_name: str
    reason: Optional[str] = None; status: str; eligible: bool; days_without_orders: int; created_at: datetime
class AdminSettingsUpdate(BaseModel):
    stripe_api_key: Optional[str] = None; shippo_api_key: Optional[str] = None
    warehouse_address: Optional[dict] = None; social_links: Optional[dict] = None

# ============== HELPERS ==============
def hash_password(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
def verify_password(p, h): return bcrypt.checkpw(p.encode(), h.encode())
def create_token(uid, email, is_admin=False):
    return jwt.encode({"sub": uid, "email": email, "is_admin": is_admin, "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if not uid: raise HTTPException(401, "Invalid token")
        user = await db.users.find_one({"id": uid})
        if not user: raise HTTPException(401, "User not found")
        return user
    except jwt.ExpiredSignatureError: raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError: raise HTTPException(401, "Invalid token")

async def require_admin(creds: HTTPAuthorizationCredentials = Depends(security)):
    user = await get_current_user(creds)
    if not user.get("is_admin"): raise HTTPException(403, "Admin access required")
    return user

async def compute_available_ml(lid, remaining_ml):
    orders = await db.orders.find({"listing_id": lid, "status": {"$nin": ["cancelled", "refunded"]}}).to_list(1000)
    return max(0, remaining_ml - sum(o.get("selected_size_ml", 0) for o in orders))

async def listing_to_response(l):
    l = clean_doc(l)
    rem = l.get("remaining_ml", l.get("full_bottle_size_ml", 0))
    avail = await compute_available_ml(l["id"], rem)
    full = l["full_bottle_size_ml"]
    fill = round((rem / full) * 100, 1) if full > 0 else 0
    return ListingResponse(id=l["id"], seller_id=l["seller_id"], seller_name=l["seller_name"],
        brand=l["brand"], name=l["name"], concentration=l["concentration"],
        full_bottle_size_ml=full, remaining_ml=rem, available_ml=avail, fill_percent=fill,
        notes=l["notes"], image=l.get("image"), batch_year=l.get("batch_year"),
        status=l["status"] if avail > 0 else "sold_out", price_per_ml=l["price_per_ml"],
        platform_fee_percent=l.get("platform_fee_percent", NORMAL_FEE),
        fee_type=l.get("fee_type", "normal"),
        authenticity_status=l.get("authenticity_status", "verified"),
        authenticity_note=l.get("authenticity_note"), created_at=l["created_at"])

def user_response(u):
    return UserResponse(id=u["id"], email=u["email"], name=u["name"],
        is_admin=u.get("is_admin", False), balance=u.get("balance", 0.0), created_at=u["created_at"])

async def get_launch_mode():
    s = await db.settings.find_one({"key": "launch_mode"})
    if not s:
        await db.settings.insert_one({"key": "launch_mode", "value": "seller_only", "updated_at": datetime.utcnow()})
        return "seller_only"
    return s["value"]

async def require_full_mode():
    if await get_launch_mode() == "seller_only":
        raise HTTPException(403, "Marketplace not yet open.")

async def get_setting(key, default=None):
    s = await db.settings.find_one({"key": key})
    return s["value"] if s else default

async def run_authenticity_check(brand, name, concentration, price_per_ml, notes, has_image):
    try:
        chat = LlmChat(api_key=LLM_KEY, session_id=f"auth-{uuid.uuid4()}", system_message=(
            'Fragrance authenticity expert. Respond ONLY with JSON: {"status":"verified" or "flagged","confidence":0-100,"reason":"brief explanation"}. '
            'Flag if: price suspicious for brand, description generic/copied, brand misspelled, no image for luxury brand.'
        )).with_model("openai", "gpt-5.2")
        resp = await chat.send_message(UserMessage(text=f"Brand:{brand} Name:{name} Conc:{concentration} ${price_per_ml}/ml Desc:{notes or 'None'} HasImage:{has_image}"))
        text = resp.strip()
        if "```" in text: text = text.split("```")[1]; text = text[4:] if text.startswith("json") else text
        data = jsonlib.loads(text)
        return {"status": data.get("status", "verified"), "confidence": data.get("confidence", 80), "reason": data.get("reason", "")}
    except Exception as e:
        logger.error(f"AI auth check: {e}")
        return {"status": "verified", "confidence": 50, "reason": "Auto-verified (AI unavailable)"}

# ============== SEED ==============
@app.on_event("startup")
async def seed_admin():
    existing = await db.users.find_one({"email": ADMIN_EMAIL})
    if not existing:
        await db.users.insert_one({"id": str(uuid.uuid4()), "email": ADMIN_EMAIL,
            "password_hash": hash_password(ADMIN_PASSWORD_RAW), "name": ADMIN_NAME,
            "is_admin": True, "balance": 0.0, "created_at": datetime.utcnow()})

# ============== AUTH ==============
@api_router.post("/auth/register", response_model=TokenResponse)
async def register(d: UserCreate):
    if d.email.lower() == ADMIN_EMAIL.lower(): raise HTTPException(400, "Email reserved")
    if await db.users.find_one({"email": d.email}): raise HTTPException(400, "Email already registered")
    uid = str(uuid.uuid4())
    user = {"id": uid, "email": d.email, "password_hash": hash_password(d.password), "name": d.name, "is_admin": False, "balance": 0.0, "created_at": datetime.utcnow()}
    await db.users.insert_one(user)
    return TokenResponse(access_token=create_token(uid, d.email), user=user_response(user))

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(d: UserLogin):
    user = await db.users.find_one({"email": d.email})
    if not user or not verify_password(d.password, user["password_hash"]): raise HTTPException(401, "Invalid credentials")
    return TokenResponse(access_token=create_token(user["id"], user["email"], user.get("is_admin", False)), user=user_response(user))

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(u=Depends(get_current_user)):
    return user_response(u)

# ============== LAUNCH MODE ==============
@api_router.get("/launch-mode")
async def check_launch_mode():
    return {"launch_mode": await get_launch_mode()}

@api_router.post("/launch-mode")
async def set_launch_mode(mode: str, u=Depends(require_admin)):
    if mode not in ("seller_only", "full"): raise HTTPException(400, "Invalid mode")
    await db.settings.update_one({"key": "launch_mode"}, {"$set": {"value": mode, "updated_at": datetime.utcnow()}}, upsert=True)
    return {"launch_mode": mode}

# ============== LISTINGS ==============
# CRITICAL: AI check → ALWAYS pending_review → Admin must approve manually
@api_router.post("/listings", response_model=ListingResponse)
async def create_listing(listing: ListingCreate, u=Depends(get_current_user)):
    lid = str(uuid.uuid4())
    ai_result = await run_authenticity_check(listing.brand, listing.name, listing.concentration, listing.price_per_ml, listing.notes, listing.image is not None)
    mode = await get_launch_mode()
    fee_pct = EARLY_ACCESS_FEE if mode == "seller_only" else NORMAL_FEE
    fee_type = "early_access" if mode == "seller_only" else "normal"
    # ALL listings go to pending_review — admin must approve manually
    doc = {"id": lid, "seller_id": u["id"], "seller_name": u["name"], "brand": listing.brand, "name": listing.name,
        "concentration": listing.concentration, "full_bottle_size_ml": listing.full_bottle_size_ml,
        "remaining_ml": listing.remaining_ml, "notes": listing.notes,
        "image": listing.image, "batch_year": listing.batch_year,
        "status": "pending_review",  # ALWAYS pending — admin must approve
        "price_per_ml": listing.price_per_ml, "platform_fee_percent": fee_pct, "fee_type": fee_type,
        "authenticity_status": ai_result["status"],
        "authenticity_note": ai_result.get("reason", ""),
        "authenticity_confidence": ai_result.get("confidence", 0),
        "created_at": datetime.utcnow()}
    await db.listings.insert_one(doc)
    return await listing_to_response(doc)

@api_router.put("/listings/{lid}", response_model=ListingResponse)
async def update_listing(lid: str, upd: ListingUpdate, u=Depends(get_current_user)):
    l = await db.listings.find_one({"id": lid})
    if not l: raise HTTPException(404, "Not found")
    if l["seller_id"] != u["id"] and not u.get("is_admin"): raise HTTPException(403, "Not authorized")
    fields = {}
    if upd.notes is not None: fields["notes"] = upd.notes
    if upd.price_per_ml is not None: fields["price_per_ml"] = upd.price_per_ml
    if upd.remaining_ml is not None: fields["remaining_ml"] = upd.remaining_ml
    if upd.image is not None: fields["image"] = upd.image
    if upd.status and upd.status in ("active", "paused"): fields["status"] = upd.status
    if fields: await db.listings.update_one({"id": lid}, {"$set": fields})
    return await listing_to_response(await db.listings.find_one({"id": lid}))

@api_router.delete("/listings/{lid}")
async def delete_listing(lid: str, u=Depends(get_current_user)):
    l = await db.listings.find_one({"id": lid})
    if not l: raise HTTPException(404, "Not found")
    if l["seller_id"] != u["id"] and not u.get("is_admin"): raise HTTPException(403, "Not authorized")
    await db.listings.update_one({"id": lid}, {"$set": {"status": "inactive"}})
    return {"message": "Deactivated"}

@api_router.get("/listings", response_model=List[ListingResponse])
async def get_listings(status: Optional[str] = None, search: Optional[str] = None):
    q = {}
    if status: q["status"] = status
    if search: q["$or"] = [{"brand": {"$regex": search, "$options": "i"}}, {"name": {"$regex": search, "$options": "i"}}]
    return [await listing_to_response(l) for l in await db.listings.find(q).sort("created_at", -1).to_list(100)]

@api_router.get("/listings/{lid}", response_model=ListingResponse)
async def get_listing(lid: str):
    l = await db.listings.find_one({"id": lid})
    if not l: raise HTTPException(404, "Not found")
    return await listing_to_response(l)

@api_router.get("/my-listings", response_model=List[ListingResponse])
async def get_my_listings(u=Depends(get_current_user)):
    return [await listing_to_response(l) for l in await db.listings.find({"seller_id": u["id"]}).sort("created_at", -1).to_list(100)]

# ============== SAVED & RECENTLY VIEWED ==============
@api_router.post("/saved-listings/{lid}")
async def save_listing(lid: str, u=Depends(get_current_user)):
    if await db.saved_listings.find_one({"user_id": u["id"], "listing_id": lid}): return {"message": "Already saved"}
    await db.saved_listings.insert_one({"id": str(uuid.uuid4()), "user_id": u["id"], "listing_id": lid, "created_at": datetime.utcnow()})
    return {"message": "Saved"}

@api_router.delete("/saved-listings/{lid}")
async def unsave_listing(lid: str, u=Depends(get_current_user)):
    await db.saved_listings.delete_one({"user_id": u["id"], "listing_id": lid})
    return {"message": "Removed"}

@api_router.get("/saved-listings")
async def get_saved_listings(u=Depends(get_current_user)):
    result = []
    for s in await db.saved_listings.find({"user_id": u["id"]}).sort("created_at", -1).to_list(100):
        l = await db.listings.find_one({"id": s["listing_id"]})
        if l: result.append(await listing_to_response(l))
    return result

@api_router.get("/saved-listings/ids")
async def get_saved_ids(u=Depends(get_current_user)):
    saved = await db.saved_listings.find({"user_id": u["id"]}).to_list(200)
    return {"ids": [s["listing_id"] for s in saved]}

@api_router.post("/recently-viewed/{lid}")
async def record_view(lid: str, u=Depends(get_current_user)):
    await db.recently_viewed.delete_one({"user_id": u["id"], "listing_id": lid})
    await db.recently_viewed.insert_one({"id": str(uuid.uuid4()), "user_id": u["id"], "listing_id": lid, "viewed_at": datetime.utcnow()})
    # Keep only last 20
    views = await db.recently_viewed.find({"user_id": u["id"]}).sort("viewed_at", -1).to_list(100)
    if len(views) > 20:
        old_ids = [v["id"] for v in views[20:]]
        await db.recently_viewed.delete_many({"id": {"$in": old_ids}})
    return {"message": "Recorded"}

@api_router.get("/recently-viewed")
async def get_recently_viewed(u=Depends(get_current_user)):
    result = []
    for v in await db.recently_viewed.find({"user_id": u["id"]}).sort("viewed_at", -1).to_list(20):
        l = await db.listings.find_one({"id": v["listing_id"]})
        if l: result.append(await listing_to_response(l))
    return result

# ============== STRIPE CHECKOUT ==============
@api_router.post("/checkout")
async def create_checkout(d: CheckoutRequest, request: Request):
    await require_full_mode()
    if d.selected_size_ml not in [5, 10]: raise HTTPException(400, "Size must be 5 or 10")
    listing = await db.listings.find_one({"id": d.listing_id})
    if not listing: raise HTTPException(404, "Not found")
    if listing["status"] != "active": raise HTTPException(400, "Not active")
    rem = listing.get("remaining_ml", listing.get("full_bottle_size_ml", 0))
    avail = await compute_available_ml(listing["id"], rem)
    if avail < d.selected_size_ml: raise HTTPException(400, f"Only {avail}ml available")
    product_price = round(listing["price_per_ml"] * d.selected_size_ml, 2)
    fee_rate = listing.get("platform_fee_percent", NORMAL_FEE)
    platform_fee = round(product_price * fee_rate, 2)
    total = round(product_price + platform_fee + SHIPPING_FEE_BUYER, 2)
    seller_net = round(product_price - platform_fee, 2)
    oid = str(uuid.uuid4())
    await db.orders.insert_one({"id": oid, "listing_id": listing["id"], "listing_brand": listing["brand"],
        "listing_name": listing["name"], "listing_concentration": listing["concentration"],
        "listing_image": listing.get("image"), "seller_id": listing["seller_id"],
        "seller_name": listing["seller_name"], "buyer_name": d.buyer_name, "buyer_email": d.buyer_email,
        "selected_size_ml": d.selected_size_ml, "price_paid": total, "seller_earnings": seller_net,
        "status": "awaiting_payment", "created_at": datetime.utcnow()})
    origin = d.origin_url.rstrip("/")
    stripe_key = await get_setting("stripe_api_key", STRIPE_KEY)
    sc = StripeCheckout(api_key=stripe_key, webhook_url=f"{str(request.base_url)}api/webhook/stripe")
    session = await sc.create_checkout_session(CheckoutSessionRequest(
        amount=float(total), currency="usd",
        success_url=f"{origin}/payment-success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{origin}/payment-cancel?order_id={oid}",
        metadata={"order_id": oid, "listing_id": listing["id"], "seller_id": listing["seller_id"]}))
    await db.payment_transactions.insert_one({"id": str(uuid.uuid4()), "order_id": oid,
        "stripe_session_id": session.session_id, "listing_id": listing["id"],
        "listing_brand": listing["brand"], "listing_name": listing["name"],
        "seller_id": listing["seller_id"], "seller_name": listing["seller_name"],
        "buyer_name": d.buyer_name, "buyer_email": d.buyer_email,
        "selected_size_ml": d.selected_size_ml, "product_price": product_price,
        "platform_fee": platform_fee, "shipping_fee": SHIPPING_FEE_BUYER, "total_amount": total,
        "seller_net_earnings": seller_net, "payment_status": "initiated",
        "payout_status": "pending", "stripe_transfer_id": None, "created_at": datetime.utcnow()})
    return {"checkout_url": session.url, "session_id": session.session_id, "order_id": oid,
        "breakdown": {"product_price": product_price, "platform_fee": platform_fee, "fee_rate": fee_rate,
            "shipping_fee": SHIPPING_FEE_BUYER, "total": total, "seller_net_earnings": seller_net}}

@api_router.get("/checkout/status/{session_id}")
async def get_checkout_status(session_id: str, request: Request):
    pt = await db.payment_transactions.find_one({"stripe_session_id": session_id})
    if not pt: raise HTTPException(404, "Not found")
    if pt.get("payment_status") == "paid": return PaymentTransactionResponse(**clean_doc(pt))
    stripe_key = await get_setting("stripe_api_key", STRIPE_KEY)
    sc = StripeCheckout(api_key=stripe_key, webhook_url=f"{str(request.base_url)}api/webhook/stripe")
    status = await sc.get_checkout_status(session_id)
    if status.payment_status == "paid" and pt.get("payment_status") != "paid":
        await db.payment_transactions.update_one({"stripe_session_id": session_id}, {"$set": {"payment_status": "paid"}})
        await db.orders.update_one({"id": pt["order_id"]}, {"$set": {"status": "pending"}})
    elif status.status == "expired":
        await db.payment_transactions.update_one({"stripe_session_id": session_id}, {"$set": {"payment_status": "expired"}})
        await db.orders.update_one({"id": pt["order_id"]}, {"$set": {"status": "cancelled"}})
    return PaymentTransactionResponse(**clean_doc(await db.payment_transactions.find_one({"stripe_session_id": session_id})))

@api_router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    body = await request.body(); sig = request.headers.get("Stripe-Signature", "")
    try:
        sc = StripeCheckout(api_key=await get_setting("stripe_api_key", STRIPE_KEY), webhook_url=f"{str(request.base_url)}api/webhook/stripe")
        event = await sc.handle_webhook(body, sig)
        if event.payment_status == "paid":
            pt = await db.payment_transactions.find_one({"stripe_session_id": event.session_id})
            if pt and pt.get("payment_status") != "paid":
                await db.payment_transactions.update_one({"stripe_session_id": event.session_id}, {"$set": {"payment_status": "paid"}})
                await db.orders.update_one({"id": pt["order_id"]}, {"$set": {"status": "pending"}})
        return {"status": "ok"}
    except Exception as e: logger.error(f"Webhook: {e}"); return {"status": "error"}

# ============== ORDERS ==============
@api_router.get("/orders/track", response_model=List[OrderResponseV2])
async def track_orders(email: str):
    results = []
    for o in await db.orders.find({"buyer_email": email}).sort("created_at", -1).to_list(100):
        try: results.append(OrderResponseV2(**clean_doc(o)))
        except: pass
    return results

@api_router.get("/orders/{oid}", response_model=OrderResponseV2)
async def get_order(oid: str):
    o = await db.orders.find_one({"id": oid})
    if not o: raise HTTPException(404, "Not found")
    return OrderResponseV2(**clean_doc(o))

@api_router.post("/orders/{oid}/confirm-delivery")
async def confirm_delivery(oid: str):
    await db.orders.update_one({"id": oid}, {"$set": {"status": "delivered"}})
    await db.payment_transactions.update_one({"order_id": oid}, {"$set": {"payout_status": "eligible"}})
    return {"message": "Delivery confirmed"}

@api_router.post("/payouts/{oid}/release")
async def release_payout(oid: str, u=Depends(get_current_user)):
    pt = await db.payment_transactions.find_one({"order_id": oid})
    if not pt: raise HTTPException(404, "Not found")
    if pt["payout_status"] != "eligible": raise HTTPException(400, "Not eligible")
    tid = f"tr_{uuid.uuid4().hex[:16]}"
    await db.payment_transactions.update_one({"order_id": oid}, {"$set": {"payout_status": "paid_out", "stripe_transfer_id": tid}})
    await db.orders.update_one({"id": oid}, {"$set": {"status": "completed"}})
    await db.users.update_one({"id": pt["seller_id"]}, {"$inc": {"balance": pt["seller_net_earnings"]}})
    return {"payout_status": "paid_out", "stripe_transfer_id": tid}

@api_router.get("/my-payouts")
async def get_my_payouts(u=Depends(get_current_user)):
    return [PaymentTransactionResponse(**clean_doc(p)) for p in await db.payment_transactions.find({"seller_id": u["id"]}).sort("created_at", -1).to_list(100)]

# ============== LABEL REQUESTS ==============
@api_router.post("/label-requests", response_model=LabelRequestResponse)
async def request_label(d: LabelRequestCreate, u=Depends(get_current_user)):
    order = await db.orders.find_one({"id": d.order_id})
    if not order: raise HTTPException(404, "Order not found")
    if order["seller_id"] != u["id"]: raise HTTPException(403, "Not your order")
    if await db.label_requests.find_one({"order_id": d.order_id, "status": {"$in": ["pending", "approved"]}}):
        raise HTTPException(400, "Already requested")
    await db.users.update_one({"id": u["id"]}, {"$inc": {"balance": -LABEL_FEE}})
    lid = str(uuid.uuid4())
    doc = {"id": lid, "order_id": d.order_id, "seller_id": u["id"], "seller_name": u["name"],
        "seller_address": d.seller_address, "listing_brand": order.get("listing_brand", ""),
        "listing_name": order.get("listing_name", ""), "buyer_name": order.get("buyer_name", ""),
        "status": "pending", "label_url": None, "tracking_number": None, "carrier": None,
        "label_fee_status": "paid", "label_fee_amount": LABEL_FEE, "created_at": datetime.utcnow()}
    await db.label_requests.insert_one(doc)
    return LabelRequestResponse(**clean_doc(doc))

@api_router.get("/my-label-requests")
async def get_my_label_requests(u=Depends(get_current_user)):
    return [LabelRequestResponse(**clean_doc(r)) for r in await db.label_requests.find({"seller_id": u["id"]}).sort("created_at", -1).to_list(50)]

# ============== MESSAGING ==============
@api_router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(d: ConversationCreate, u=Depends(get_current_user)):
    if d.recipient_id == u["id"]: raise HTTPException(400, "Cannot message yourself")
    recip = await db.users.find_one({"id": d.recipient_id})
    if not recip: raise HTTPException(404, "User not found")
    q = {"participants": {"$all": [u["id"], d.recipient_id]}, "listing_id": d.listing_id or None}
    existing = await db.conversations.find_one(q)
    if existing:
        c = clean_doc(existing)
        c["unread_count"] = await db.messages.count_documents({"conversation_id": c["id"], "sender_id": {"$ne": u["id"]}, "read": False})
        return ConversationResponse(**c)
    cid = str(uuid.uuid4()); ln, lb = None, None
    if d.listing_id:
        ll = await db.listings.find_one({"id": d.listing_id})
        if ll: ln, lb = ll["name"], ll["brand"]
    doc = {"id": cid, "participants": [u["id"], d.recipient_id], "participant_names": {u["id"]: u["name"], d.recipient_id: recip["name"]},
        "listing_id": d.listing_id, "listing_name": ln, "listing_brand": lb, "last_message": "", "last_message_at": None, "created_at": datetime.utcnow()}
    await db.conversations.insert_one(doc); clean_doc(doc); doc["unread_count"] = 0
    return ConversationResponse(**doc)

@api_router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(u=Depends(get_current_user)):
    result = []
    for c in await db.conversations.find({"participants": u["id"]}).sort("last_message_at", -1).to_list(100):
        c = clean_doc(c)
        c["unread_count"] = await db.messages.count_documents({"conversation_id": c["id"], "sender_id": {"$ne": u["id"]}, "read": False})
        result.append(ConversationResponse(**c))
    return result

@api_router.get("/conversations/unread-count")
async def unread_count(u=Depends(get_current_user)):
    total = 0
    for c in await db.conversations.find({"participants": u["id"]}).to_list(100):
        total += await db.messages.count_documents({"conversation_id": c["id"], "sender_id": {"$ne": u["id"]}, "read": False})
    return {"unread_count": total}

@api_router.get("/conversations/{cid}/messages", response_model=List[MessageResponse])
async def get_messages(cid: str, u=Depends(get_current_user)):
    c = await db.conversations.find_one({"id": cid})
    if not c or u["id"] not in c["participants"]: raise HTTPException(404, "Not found")
    await db.messages.update_many({"conversation_id": cid, "sender_id": {"$ne": u["id"]}, "read": False}, {"$set": {"read": True}})
    return [MessageResponse(**clean_doc(m)) for m in await db.messages.find({"conversation_id": cid}).sort("created_at", 1).to_list(500)]

@api_router.post("/conversations/{cid}/messages", response_model=MessageResponse)
async def send_message(cid: str, d: MessageCreate, u=Depends(get_current_user)):
    c = await db.conversations.find_one({"id": cid})
    if not c or u["id"] not in c["participants"]: raise HTTPException(404, "Not found")
    mid = str(uuid.uuid4())
    doc = {"id": mid, "conversation_id": cid, "sender_id": u["id"], "sender_name": u["name"], "text": d.text.strip(), "read": False, "created_at": datetime.utcnow()}
    await db.messages.insert_one(doc); clean_doc(doc)
    await db.conversations.update_one({"id": cid}, {"$set": {"last_message": d.text.strip()[:100], "last_message_at": doc["created_at"]}})
    return MessageResponse(**doc)

# ============== RETURNS ==============
@api_router.get("/returns/eligibility/{lid}")
async def check_return_eligibility(lid: str, u=Depends(get_current_user)):
    l = await db.listings.find_one({"id": lid, "seller_id": u["id"]})
    if not l: raise HTTPException(404, "Not found")
    last_order = await db.orders.find_one({"listing_id": lid, "status": {"$nin": ["cancelled"]}}, sort=[("created_at", -1)])
    days = (datetime.utcnow() - (last_order["created_at"] if last_order else l["created_at"])).days
    return {"eligible": days >= 30, "days_without_orders": days}

@api_router.post("/returns", response_model=ReturnResponse)
async def request_return(d: ReturnRequestCreate, u=Depends(get_current_user)):
    l = await db.listings.find_one({"id": d.listing_id, "seller_id": u["id"]})
    if not l: raise HTTPException(404, "Not found")
    last_order = await db.orders.find_one({"listing_id": d.listing_id, "status": {"$nin": ["cancelled"]}}, sort=[("created_at", -1)])
    days = (datetime.utcnow() - (last_order["created_at"] if last_order else l["created_at"])).days
    if days < 30: raise HTTPException(400, f"Not eligible — {days} days")
    if await db.returns.find_one({"listing_id": d.listing_id, "status": {"$in": ["pending", "approved"]}}): raise HTTPException(400, "Already requested")
    rid = str(uuid.uuid4())
    doc = {"id": rid, "seller_id": u["id"], "listing_id": d.listing_id, "listing_brand": l["brand"], "listing_name": l["name"],
        "reason": d.reason, "status": "pending", "eligible": True, "days_without_orders": days, "created_at": datetime.utcnow()}
    await db.returns.insert_one(doc)
    return ReturnResponse(**clean_doc(doc))

@api_router.get("/my-returns")
async def get_my_returns(u=Depends(get_current_user)):
    return [ReturnResponse(**clean_doc(r)) for r in await db.returns.find({"seller_id": u["id"]}).sort("created_at", -1).to_list(50)]

# ============== FAQ ==============
@api_router.get("/faq")
async def get_faq():
    social = await get_setting("social_links", {})
    mode = await get_launch_mode()
    ea_fee = "1%"
    normal_fee = "10%"
    return {
        "support_email": "supportcolognemarket@gmail.com",
        "social_links": social or {},
        "questions": [
            {"q": "How does Cologne Market work?", "a": "Sellers list their cologne bottles. After AI verification and admin approval, listings go live. Buyers purchase 5ml or 10ml decants. Sellers ship bottles to our facility where we decant and fulfill orders."},
            {"q": "What are the seller fees?", "a": f"During Early Access: {ea_fee} platform fee (you keep 99%). During Normal mode: {normal_fee} platform fee (you keep 90%). Fees are locked when the listing is created."},
            {"q": "How much does shipping cost?", "a": "Buyers pay a flat $5 shipping fee per order. Sellers pay a refundable $5 fee for inbound shipping labels. The $5 seller fee is refunded once the package is confirmed delivered to our facility."},
            {"q": "How does the AI verification work?", "a": "When you create a listing, our AI analyzes your product details for authenticity. After AI review, every listing requires manual admin approval before going live. No listing bypasses this process."},
            {"q": "How do I get paid as a seller?", "a": "After a buyer confirms delivery, your earnings become eligible for payout. Earnings are credited to your Cologne Market balance. Your net earnings = product price minus the platform fee."},
            {"q": "What decant sizes are available?", "a": "We offer 5ml and 10ml decants. The price is calculated as price per ml multiplied by the decant size."},
            {"q": "Can I return my bottle if it doesn't sell?", "a": "Yes. If your listing receives no orders for 30 days, you can request a return. Returns require admin approval."},
            {"q": "How do I create a listing?", "a": "Tap the Sell (+) tab, fill in your cologne details including brand, name, full bottle size, remaining ml, price per ml, and description. Upload a photo for best results."},
            {"q": "What is Early Access mode?", "a": "Early Access is our launch phase where sellers can sign up and list products at a reduced 1% fee. The marketplace opens to buyers when admin switches to Full Access mode."},
            {"q": "How do I track my order?", "a": "Go to Dashboard > Orders section and enter the email you used at checkout. You'll see all your orders with current status."},
        ]
    }

# ============== DASHBOARD ==============
@api_router.get("/dashboard/stats")
async def dashboard_stats(u=Depends(get_current_user)):
    sid = u["id"]
    orders = await db.orders.find({"seller_id": sid, "status": {"$nin": ["cancelled", "refunded"]}}).to_list(1000)
    return {"total_listings": await db.listings.count_documents({"seller_id": sid}),
        "active_listings": await db.listings.count_documents({"seller_id": sid, "status": "active"}),
        "total_sold": len(orders), "total_earnings": round(sum(o.get("seller_earnings", 0) for o in orders), 2),
        "balance": u.get("balance", 0.0)}

@api_router.get("/dashboard/orders")
async def seller_orders(u=Depends(get_current_user)):
    results = []
    for o in await db.orders.find({"seller_id": u["id"]}).sort("created_at", -1).to_list(100):
        try: results.append(OrderResponseV2(**clean_doc(o)))
        except: pass
    return results

# ============== ADMIN (ISOLATED) ==============
@api_router.get("/admin/metrics")
async def admin_metrics(u=Depends(require_admin)):
    pts = await db.payment_transactions.find({"payment_status": "paid"}).to_list(5000)
    total_revenue = sum(p.get("total_amount", 0) for p in pts)
    total_platform_fees = sum(p.get("platform_fee", 0) for p in pts)
    total_shipping = sum(p.get("shipping_fee", 0) for p in pts)
    platform_profit = total_platform_fees + total_shipping
    refund_amount = sum(o.get("price_paid", 0) for o in await db.orders.find({"status": "refunded"}).to_list(500))
    monthly = {}
    for p in pts:
        key = p["created_at"].strftime("%Y-%m")
        if key not in monthly: monthly[key] = {"revenue": 0, "profit": 0, "orders": 0}
        monthly[key]["revenue"] += p.get("total_amount", 0)
        monthly[key]["profit"] += p.get("platform_fee", 0) + p.get("shipping_fee", 0)
        monthly[key]["orders"] += 1
    return {
        "financial": {"total_revenue": round(total_revenue, 2), "platform_profit": round(platform_profit, 2),
            "net_earnings": round(platform_profit - refund_amount, 2),
            "avg_order_value": round(total_revenue / len(pts), 2) if pts else 0},
        "platform": {"total_orders": await db.orders.count_documents({}),
            "active_users": await db.users.count_documents({"is_admin": {"$ne": True}}),
            "total_listings": await db.listings.count_documents({}),
            "active_listings": await db.listings.count_documents({"status": "active"}),
            "pending_review": await db.listings.count_documents({"status": "pending_review"}),
            "early_access_listings": await db.listings.count_documents({"fee_type": "early_access"}),
            "normal_listings": await db.listings.count_documents({"fee_type": "normal"}),
            "flagged_listings": await db.listings.count_documents({"authenticity_status": "flagged"}),
            "pending_labels": await db.label_requests.count_documents({"status": "pending"}),
            "pending_returns": await db.returns.count_documents({"status": "pending"})},
        "monthly": [{"month": k, **{kk: round(vv, 2) for kk, vv in v.items()}} for k, v in sorted(monthly.items())],
        "launch_mode": await get_launch_mode()
    }

@api_router.get("/admin/orders")
async def admin_orders(status: Optional[str] = None, u=Depends(require_admin)):
    q = {"status": status} if status else {}
    results = []
    for o in await db.orders.find(q).sort("created_at", -1).to_list(200):
        try: results.append(OrderResponseV2(**clean_doc(o)))
        except: pass
    return results

@api_router.get("/admin/listings")
async def admin_listings(status: Optional[str] = None, u=Depends(require_admin)):
    q = {"status": status} if status else {}
    return [await listing_to_response(l) for l in await db.listings.find(q).sort("created_at", -1).to_list(200)]

@api_router.get("/admin/pending-review")
async def admin_pending_review(u=Depends(require_admin)):
    """All listings awaiting admin approval (AI checked but need manual approve)."""
    return [await listing_to_response(l) for l in await db.listings.find({"status": "pending_review"}).sort("created_at", -1).to_list(100)]

@api_router.get("/admin/flagged")
async def admin_flagged(u=Depends(require_admin)):
    """Flagged + pending_review listings."""
    return [await listing_to_response(l) for l in await db.listings.find({"$or": [{"authenticity_status": "flagged"}, {"status": "pending_review"}]}).sort("created_at", -1).to_list(100)]

@api_router.post("/admin/listings/{lid}/approve")
async def admin_approve(lid: str, u=Depends(require_admin)):
    await db.listings.update_one({"id": lid}, {"$set": {"authenticity_status": "verified", "status": "active"}})
    return {"message": "Listing approved and live"}

@api_router.post("/admin/listings/{lid}/reject")
async def admin_reject(lid: str, u=Depends(require_admin)):
    await db.listings.update_one({"id": lid}, {"$set": {"authenticity_status": "rejected", "status": "inactive"}})
    return {"message": "Listing rejected"}

@api_router.post("/admin/orders/{oid}/refund")
async def admin_refund(oid: str, u=Depends(require_admin)):
    await db.orders.update_one({"id": oid}, {"$set": {"status": "refunded"}})
    await db.payment_transactions.update_one({"order_id": oid}, {"$set": {"payment_status": "refunded", "payout_status": "refunded"}})
    return {"message": "Refunded"}

@api_router.get("/admin/returns")
async def admin_returns(u=Depends(require_admin)):
    return [ReturnResponse(**clean_doc(r)) for r in await db.returns.find({}).sort("created_at", -1).to_list(100)]

@api_router.post("/admin/returns/{rid}/approve")
async def admin_approve_return(rid: str, u=Depends(require_admin)):
    await db.returns.update_one({"id": rid}, {"$set": {"status": "approved"}})
    r = await db.returns.find_one({"id": rid})
    if r: await db.listings.update_one({"id": r["listing_id"]}, {"$set": {"status": "inactive"}})
    return {"message": "Return approved"}

@api_router.post("/admin/returns/{rid}/deny")
async def admin_deny_return(rid: str, u=Depends(require_admin)):
    await db.returns.update_one({"id": rid}, {"$set": {"status": "denied"}})
    return {"message": "Denied"}

@api_router.get("/admin/label-requests")
async def admin_label_requests(status: Optional[str] = None, u=Depends(require_admin)):
    q = {"status": status} if status else {}
    return [LabelRequestResponse(**clean_doc(r)) for r in await db.label_requests.find(q).sort("created_at", -1).to_list(100)]

@api_router.post("/admin/label-requests/{lid}/approve")
async def admin_approve_label(lid: str, u=Depends(require_admin)):
    lr = await db.label_requests.find_one({"id": lid})
    if not lr: raise HTTPException(404, "Not found")
    tracking = f"USPS{uuid.uuid4().hex[:12].upper()}"
    await db.label_requests.update_one({"id": lid}, {"$set": {"status": "approved", "tracking_number": tracking,
        "label_url": f"https://labels.colognemarket.com/{lid[:8]}.pdf", "carrier": "USPS Ground Advantage"}})
    await db.orders.update_one({"id": lr["order_id"]}, {"$set": {"status": "label_generated"}})
    return {"message": "Label generated (ground-only domestic)", "tracking_number": tracking}

@api_router.post("/admin/label-requests/{lid}/reject")
async def admin_reject_label(lid: str, u=Depends(require_admin)):
    lr = await db.label_requests.find_one({"id": lid})
    if not lr: raise HTTPException(404, "Not found")
    await db.label_requests.update_one({"id": lid}, {"$set": {"status": "rejected"}})
    await db.users.update_one({"id": lr["seller_id"]}, {"$inc": {"balance": LABEL_FEE}})
    await db.label_requests.update_one({"id": lid}, {"$set": {"label_fee_status": "refunded"}})
    return {"message": "Rejected, $5 refunded"}

@api_router.post("/admin/label-requests/{lid}/confirm-delivery")
async def admin_confirm_label_delivery(lid: str, u=Depends(require_admin)):
    lr = await db.label_requests.find_one({"id": lid})
    if not lr: raise HTTPException(404, "Not found")
    if lr.get("label_fee_status") == "refunded": raise HTTPException(400, "Already refunded")
    await db.users.update_one({"id": lr["seller_id"]}, {"$inc": {"balance": LABEL_FEE}})
    await db.label_requests.update_one({"id": lid}, {"$set": {"label_fee_status": "refunded", "status": "delivered"}})
    return {"message": "$5 refunded"}

@api_router.get("/admin/settings")
async def admin_get_settings(u=Depends(require_admin)):
    sk = await get_setting("stripe_api_key"); sp = await get_setting("shippo_api_key")
    return {"stripe_api_key": f"***{sk[-4:]}" if sk else None, "stripe_configured": bool(sk),
        "shippo_api_key": f"***{sp[-4:]}" if sp else None, "shippo_configured": bool(sp),
        "warehouse_address": await get_setting("warehouse_address"),
        "social_links": await get_setting("social_links", {})}

@api_router.post("/admin/settings")
async def admin_update_settings(d: AdminSettingsUpdate, u=Depends(require_admin)):
    if d.stripe_api_key is not None:
        await db.settings.update_one({"key": "stripe_api_key"}, {"$set": {"value": d.stripe_api_key}}, upsert=True)
    if d.shippo_api_key is not None:
        await db.settings.update_one({"key": "shippo_api_key"}, {"$set": {"value": d.shippo_api_key}}, upsert=True)
    if d.warehouse_address is not None:
        await db.settings.update_one({"key": "warehouse_address"}, {"$set": {"value": d.warehouse_address}}, upsert=True)
    if d.social_links is not None:
        await db.settings.update_one({"key": "social_links"}, {"$set": {"value": d.social_links}}, upsert=True)
    return {"message": "Settings updated"}

# ============== APP CONFIG ==============
@api_router.get("/app-config")
async def app_config():
    """Public config for the app."""
    return {"app_name": "Cologne Market", "logo_url": LOGO_URL,
        "support_email": "supportcolognemarket@gmail.com",
        "launch_mode": await get_launch_mode()}

# ============== UTILITY ==============
@api_router.get("/")
async def root():
    return {"message": "Cologne Market API", "version": "9.0"}

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("shutdown")
async def shutdown():
    client.close()
